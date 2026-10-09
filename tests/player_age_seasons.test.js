// Run with: node --test tests/player_age_seasons.test.js
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('lol_ticker/player_rankings.html', 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
// Exercise the actual page's observation builder without loading its DOM or API.
const context = vm.createContext({});
vm.runInContext(`
  const finite = value => value != null && value !== '' && Number.isFinite(Number(value));
  const num = value => finite(value) ? Number(value) : null;
  const ROLE_COLORS = {top:1,jungle:1,mid:1,bot:1,support:1,unknown:1};
  ${script.match(/^function roleKey.*$/m)[0]}
  ${script.match(/^function exactAge.*$/m)[0]}
  ${script.split('// Player-season observations:')[1].split('// End player-season observations.')[0].split('\n').slice(1).join('\n')}
`, context);
const rows = players => JSON.parse(JSON.stringify(context.seasonObservations(players)));
const point = (date, rating=60, extra={}) => ({date, rating, impact:100,
  effective_games:50, games:200, role:'mid', team:'Old team', league:'Old league', ...extra});
const player = history => ({player_id:'faker', name:'Faker', birthday:'1996-05-07',
  team:'Current team', league:'Current league', impact:999, ci_low:900, ci_high:1000, history});

test('one identity has separately selectable dated ages and historical teams', () => {
  const observations = rows([player([point('2015-12-31'), point('2026-10-04', 65)])]);
  assert.deepEqual(observations.map(p=>[p.player_id,p.entry_id,p.age]),
    [['faker','faker@2015',19],['faker','faker@2026',30]]);
  assert.equal(observations[0].team, 'Old team');
  assert.equal(observations[0].impact, 100);
  assert.equal(observations[0].ci_low, null);
});
test('latest eligible annual observation wins regardless of source ordering', () => {
  const observations = rows([player([point('2025-12-31',70), point('2025-04-30',60),
    point('2025-10-31',99,{effective_games:19}), point('2024-12-31',55)])]);
  assert.equal(observations.length,2);
  assert.equal(observations.find(p=>p.season==='2025').snapshot_date,'2025-12-31');
});
test('era score comes from that snapshot rather than the latest impact', () => {
  const observations = rows([player([point('2016-12-31',50),point('2017-12-31',70)])]);
  assert.deepEqual(observations.map(p=>p.era_z),[0,2]);
});
test('age changes on the birthday, with a separate observation per season', () => {
  assert.equal(context.exactAge('1996-05-07','2026-05-06'),29);
  assert.equal(context.exactAge('1996-05-07','2026-05-07'),30);
  assert.equal(context.exactAge('2000-02-29','2025-02-28'),24);
  assert.equal(context.exactAge('2000-02-29','2025-03-01'),25);
});
test('unknown birthdays stay unknown instead of borrowing current age', () => {
  const unknown = {...player([point('2019-12-31')]), birthday:null, age:25};
  assert.equal(rows([unknown])[0].age,null);
});
test('insufficient evidence and missing historical scores produce no observations', () => {
  assert.equal(rows([player([point('2018-12-31',null),
    point('2019-12-31',60,{effective_games:19.99})])]).length,0);
});
test('seasons for identical display names retain distinct source identities', () => {
  const first=player([point('2020-12-31')]), second={...first,player_id:'other-faker'};
  assert.notEqual(rows([first,second])[0].entry_id, rows([first,second])[1].entry_id);
});
