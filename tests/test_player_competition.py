"""Tests for competition attribution, transfer handling and age identity joins."""
from dataclasses import replace
from datetime import timedelta
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from lol_ticker import player_ratings as ratings
from test_player_ratings import game


class CompetitionTests(unittest.TestCase):
    def test_domestic_results_cannot_create_a_competition_offset(self):
        matches = [replace(game(n), league='Local') for n in range(40)]
        fit = ratings.fit_ratings(matches, matches[-1].played_at)
        self.assertAlmostEqual(fit.regional_strength['Local'], 0, places=8)

    def test_international_team_affiliation_does_not_follow_one_academy_substitute(self):
        a = game(0, league='Main', team_a='A', team_b='B')
        b = game(1, league='Other', team_a='C', team_b='D')
        academy = game(2, league='Academy', team_a='Sub', team_b='Reserve')
        roster = tuple(p for p in a.players if p.team=='A') + tuple(p for p in b.players if p.team=='C')
        # Preserve team identity while introducing a substitute last seen in Academy.
        roster = tuple(replace(p, side='blue' if p.team=='A' else 'red') for p in roster)
        substitute = next(p for p in academy.players if p.team=='Sub' and p.role=='top')
        roster = tuple(replace(p,player_id=substitute.player_id) if p.team=='A' and p.role=='top' else p for p in roster)
        international = replace(game(3),league='MSI',players=roster)
        contexts, _ = ratings._regional_context([a,b,academy,international])
        self.assertEqual(set(contexts[international.game_id][p.player_id] for p in roster if p.side=='blue'), {'Main'})
        self.assertEqual(set(contexts[international.game_id][p.player_id] for p in roster if p.side=='red'), {'Other'})

    def test_cross_region_results_raise_winning_competition_without_named_players(self):
        local_a = [game(n,league='RegionA',team_a='A',team_b='B') for n in range(20)]
        local_b = [game(n+20,league='RegionB',team_a='C',team_b='D') for n in range(20)]
        bridges = [replace(game(n+40,league='MSI',team_a='A',team_b='C'),gold_margin=250 if n%2==0 else -250) for n in range(50)]
        matches=local_a+local_b+bridges
        fit=ratings.fit_ratings(matches,matches[-1].played_at)
        self.assertGreater(fit.regional_strength['RegionA'],fit.regional_strength['RegionB']+20)
        contexts,_=ratings._regional_context(matches)
        matrix,_,_,_,_=ratings._design(matches,fit.index,controls=fit.controls,regions=contexts)
        np.testing.assert_allclose(ratings._predict(fit,matches),matrix@fit.raw_coefficients)
        errors,_=ratings._posterior_standard_errors(fit)
        self.assertTrue(np.all(np.isfinite(errors)))

    def test_age_uses_birthdays_instead_of_dividing_days_by_average_year(self):
        from datetime import datetime, timezone
        cutoff=datetime(2000,1,1,tzinfo=timezone.utc)
        self.assertEqual(ratings._age('1999-01-01',cutoff),1)
        self.assertLess(ratings._age('1999-01-02',cutoff),1)
        self.assertIsNone(ratings._age('2001-01-01',cutoff))

    def test_cup_and_league_renames_preserve_domestic_context(self):
        before=game(0,league='OGN')
        cup=replace(game(1),league='KeSPA Cup')
        contexts,latest=ratings._regional_context([before,cup])
        self.assertEqual(set(latest.values()),{'LCK'})
        self.assertEqual(set(contexts[cup.game_id].values()),{'LCK'})

    def test_negligible_old_evidence_stays_in_archive_but_leaves_current_solve(self):
        old=game(0,league='Old')
        recent=game(1,played_at=old.played_at+timedelta(days=1500),team_a='C',team_b='D')
        fit=ratings.fit_ratings([old,recent],recent.played_at)
        self.assertIn('A:top',fit.index)
        self.assertAlmostEqual(fit.coefficients[fit.index['A:top']],0,places=8)
        self.assertGreater(fit.weights[0],0)

    def test_canonical_page_hash_wins_over_recycled_display_names(self):
        import csv,json,tempfile,io,contextlib
        spec=importlib.util.spec_from_file_location('metadata',Path(__file__).parents[1]/'scripts/player_identity_metadata.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sources=root/'oe';sources.mkdir()
            (root/'directory.json').write_text(json.dumps({'people':[{'person_id':'Ruler'},{'person_id':'Ruler (Guillermo Torres)'}], 'aliases':[{'alias':'Ruler','entity_id':'Ruler (Guillermo Torres)','entity_type':'person','alias_type':'ign'}]}))
            (root/'births-1998.json').write_text(json.dumps([{'source_url':'https://lol.fandom.com/wiki/Ruler','birthday':'1998-12-29','name':'Ruler'}]))
            with (sources/'oe_2026.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['position','playerid']);writer.writeheader()
                for page in ['Ruler','Ruler (Guillermo Torres)']:writer.writerow({'position':'bot','playerid':module.oe_id(page)})
            with contextlib.redirect_stdout(io.StringIO()):module.build(root,sources)
            birthdays=json.loads((root/'birthdays.json').read_text())
            self.assertEqual(birthdays,{module.oe_id('Ruler'):'1998-12-29'})
            self.assertEqual(json.loads((root/'identity_aliases.json').read_text()),{})

    def test_wiki_identity_hash_is_exact_and_disambiguated(self):
        spec=importlib.util.spec_from_file_location('metadata',Path(__file__).parents[1]/'scripts/player_identity_metadata.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.oe_id('Faker'),'oe:player:e1edfc5cea461399a63cb813cf795cc')
        self.assertNotEqual(module.oe_id('Doran'),module.oe_id('Doran (Choi Hyeon-jun)'))

if __name__=='__main__':unittest.main()
