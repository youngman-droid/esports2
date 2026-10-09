const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('lol_ticker/player_rankings.html','utf8').split('<script>')[1].split('</script>')[0];
const context=vm.createContext({});
vm.runInContext(`const finite=value=>value!=null&&value!==''&&Number.isFinite(Number(value));
${script.split('// Comparison matrix:')[1].split('// End comparison matrix.')[0].split('\n').slice(1).join('\n')}`,context);
const matrix=(rows,ids,alignment='age',historical=true)=>JSON.parse(JSON.stringify(context.comparisonMatrix(rows,ids,alignment,historical)));
const row=(id,age,season)=>({player_id:id,age,season,snapshot_date:season+'-12-31',era_z:1});
test('same-age comparison aligns different seasons without inventing missing observations',()=>{
 const result=matrix([row('faker',20,'2016'),row('chovy',20,'2021'),row('faker',21,'2017')],['faker','chovy']);
 assert.deepEqual(result.map(r=>r.key),['20','21']);
 assert.equal(result[0].cells[0][0].season,'2016');
 assert.equal(result[0].cells[1][0].season,'2021');
 assert.deepEqual(result[1].cells[1],[]);
});
test('same-season comparison preserves each player age and selection order',()=>{
 const result=matrix([row('faker',25,'2021'),row('chovy',20,'2021'),row('other',30,'2020')],['chovy','faker'],'season');
 assert.equal(result.length,1);
 assert.deepEqual(result[0].cells.map(c=>c[0].age),[20,25]);
});
test('unknown ages remain separate and multiple observations at an age are retained',()=>{
 const result=matrix([row('a',null,'2020'),row('a',18,'2021'),row('a',18,'2022')],['a','b']);
 assert.deepEqual(result.map(r=>r.key),['18','Unknown age']);
 assert.equal(result[0].cells[0].length,2);
 assert.deepEqual(result[1].cells[1],[]);
});
test('current and peak views produce one side-by-side estimate row',()=>{
 const result=matrix([row('a',25,'2021'),row('b',30,'2026')],['a','b'],'age',false);
 assert.equal(result.length,1);
 assert.equal(result[0].key,'Estimate');
});
