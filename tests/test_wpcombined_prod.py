import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lol_ticker import wpgam, wpbench, wpcombined, wpcombined_prod as prod, wpcombined_live, wpx, wphist
from tests import test_wpgam as fixtures


class ProductionCombinationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root/'base.npz'
        self.joint = self.root/'joint.npz'
        self.table = self.root/'tables.npz'
        self.bundle = self.root/'bundle.json'
        self.stack = self.root/'stack.json'
        base = fixtures.ArtifactTests()._model()
        base['state']['cal_intercept'] = 0.
        base['state']['cal_slope'] = 1.
        wpgam.save_model(base, str(self.base))
        signal = np.tile([-1., 1.], 10)
        self.arrays = {k: signal[:, None] for k in ('objective', 'trend', 'composition', 'sq')}
        names = dict(objective='baron_time_adv', trend='lead_change_120s_k', composition='hp_growth_sum', sq='prior_patch_pair_score')
        specs = {k: dict(feature_names=[v], bounds=[0., 10.], l2=1., smooth=1.) for k,v in names.items()}
        specs['sq'].update(bounds=[0., 2.5], time_knots=[0,5,10,15,20], zero_after_min=20, monotone_decay=True, scale=.25)
        joint = wpcombined.fit(self.arrays, np.full(20,.5), (signal>0).astype(int), np.arange(20), np.full(20,5.), specs=specs)
        wpcombined.save(joint, self.joint)
        self.table.write_bytes(b'fixture table source')
        bundle = dict(kind=prod.BUNDLE_KIND, base=dict(path=str(self.base),sha256=prod.sha(self.base)),
                      joint=dict(path=str(self.joint),sha256=prod.sha(self.joint)), families=list(self.arrays),
                      calibration=dict(intercept=.052, slope=.985, probability_clip=[1e-5,1-1e-5]),
                      capture=dict(sq_table_path=str(self.table), sq_table_sha256=prod.sha(self.table),
                                   composition_patch_policy='gameplay_major_minor', composition_root=str(self.root)))
        self.bundle.write_text(json.dumps(bundle))
        self.write_stack()

    def write_stack(self):
        stack = dict(kind=prod.STACK_KIND,deployed=True,bundle=dict(path=str(self.bundle),sha256=prod.sha(self.bundle)))
        stack['stack_sha256'] = prod.signature(stack)
        self.stack.write_text(json.dumps(stack))

    def inputs(self):
        arrays = {k:np.array([[1.]]) for k in self.arrays}
        detail = dict(families={k:dict(available=True,names=['feature'],known=[True],reasons=['fixture']) for k in arrays},available_families=list(arrays))
        return arrays,detail

    def test_joint_before_shared_calibration_and_exact_missing(self):
        loaded = prod.load_active(self.stack)
        state = dict(t_min=5., gold_diff_k=1., gold_diff_prev_k=1.)
        draft = ['A']*5
        arrays,detail = self.inputs()
        core = wpgam.predict_live_model(loaded['base'],state,draft,draft,rounded=False)['p_blue']
        expected = wpbench._apply_platt(wpcombined.predict(loaded['joint'],arrays,[core],[5.]),loaded['calibration'])[0]
        with mock.patch.object(wpcombined_live,'blocks',return_value=(arrays,detail)):
            result = prod.predict(loaded,state,draft,draft,rounded=False)
        self.assertEqual(result['p_blue'],expected)
        self.assertEqual(result['model_kind'],prod.MODEL_KIND)
        zero = {k:np.zeros((1,1)) for k in arrays}
        with mock.patch.object(wpcombined_live,'blocks',return_value=(zero,self.inputs()[1])):
            result = prod.predict(loaded,state,draft,draft,rounded=False)
        self.assertEqual(result['p_blue'],wpbench._apply_platt([core],loaded['calibration'])[0])

    def test_no_champion_removes_both_extra_draft_channels(self):
        loaded = prod.load_active(self.stack)
        arrays,detail = self.inputs()
        with mock.patch.object(wpcombined_live,'blocks',return_value=(arrays,detail)):
            result = prod.predict(loaded,dict(t_min=5.),(),(),rounded=False)
        for k in ('composition','sq'):
            self.assertEqual(result['combination_logit'][k],0.)
            self.assertFalse(result['combination_coverage']['families'][k]['available'])
        self.assertGreater(result['combination_logit']['trend'],0.)

    def test_tampered_component_invalidates_loaded_cache(self):
        prod.load_active(self.stack)
        self.table.write_bytes(b'tampered table')
        with self.assertRaisesRegex(ValueError,'component hash'):
            prod.load_active(self.stack)

    def test_tampered_manifest_rejected(self):
        stack=json.loads(self.stack.read_text());stack['bundle']['path']='elsewhere'
        self.stack.write_text(json.dumps(stack))
        with self.assertRaisesRegex(ValueError,'checksum'):
            prod.load_active(self.stack)

    def test_production_dispatch_and_explicit_fallback(self):
        with mock.patch.object(wpx,'LIVE_MODEL_PATH',str(self.base)),mock.patch.object(wpx,'LIVE_STACK_PATH',str(self.stack)):
            result=wpx.predict_live(dict(t_min=5.),path=str(self.base))
            self.assertEqual(result['model_kind'],prod.MODEL_KIND)
            self.table.write_bytes(b'tampered')
            result=wpx.predict_live(dict(t_min=5.),path=str(self.base))
            self.assertEqual(result['model_kind'],wpgam.MODEL_KIND)
            self.assertIn('production_combination_unavailable',result['input_warnings'])

    def test_capture_pins_table_and_preserves_explicit_research_override(self):
        with mock.patch.object(prod,'load_active',return_value=prod.load_active(self.stack)):
            context=prod.capture_context({'patchVersion':'16.19.777.22'})
            self.assertEqual(context['composition_patch'],'16.19')
            self.assertEqual(context['sq_table_path'],str(self.table.resolve()))
            explicit=prod.capture_context({'patchVersion':'16.19.1'},{'sq_table_path':'different'})
            self.assertEqual(explicit['sq_table_path'],'different')

    def test_old_historical_teacher_cannot_mix_new_score(self):
        with self.assertRaisesRegex(ValueError,'production combination'):
            wphist.predict_live(.6,dict(model_kind=prod.MODEL_KIND),path='absent.npz')

    def test_rejects_disagreeing_or_invalid_game_clocks(self):
        loaded=prod.load_active(self.stack)
        for state in (dict(t_min=30.,clock_s=300.),dict(t_min=float('nan')),dict(t_min=-1.)):
            with self.subTest(state=state),self.assertRaises(ValueError):
                prod.predict(loaded,state)
        self.assertTrue(np.isfinite(prod.predict(loaded,dict(t_min=5.01,clock_s=300.))['p_blue']))


if __name__ == '__main__':
    unittest.main()


class RebaseTests(unittest.TestCase):
    def test_paths_from_a_moved_checkout_map_onto_this_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new"
            (root / "data" / "wpx").mkdir(parents=True)
            (root / "data" / "wpx" / "bundle.json").write_text("{}")
            with mock.patch.object(prod.config, "REPO_ROOT", str(root)):
                old = Path("/old/checkout/data/wpx/bundle.json")
                self.assertEqual(prod.rebase(old), root / "data" / "wpx" / "bundle.json")
                # missing here: keep the recorded path (the caller's error names it)
                missing = Path("/old/checkout/data/wpx/missing.npz")
                self.assertEqual(prod.rebase(missing), missing)
                inside = root / "data" / "wpx" / "bundle.json"
                self.assertEqual(prod.rebase(inside), inside)
                self.assertEqual(prod.rebase("relative/data/x"), Path("relative/data/x"))
                self.assertIsNone(prod.rebase(None))
