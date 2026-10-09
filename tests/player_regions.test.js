const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const html=fs.readFileSync('lol_ticker/player_rankings.html','utf8');
const context=vm.createContext({});
vm.runInContext(html.split('// Geographic home regions')[1].split('// End region resolver.')[0].split('\n').slice(1).join('\n'),context);
test('major domestic and academy circuits share a geographic region',()=>{
 for(const league of ['OGN','LCK','CK','LCKC','LAS'])assert.equal(context.circuitRegion(league),'KR');
 for(const league of ['LPL','LDL','LSPL'])assert.equal(context.circuitRegion(league),'CN');
 for(const league of ['LEC','EU LCS','LFL','LPLOL','PRM','AL','TCL'])assert.equal(context.circuitRegion(league),'EMEA');
 for(const league of ['LCS','NA LCS','NACL'])assert.equal(context.circuitRegion(league),'NA');
});
test('Asia-Pacific combines its domestic circuits while Americas preserve subdivisions',()=>{
 for(const league of ['LJL','VCS','PCS','LCP','LCO','OPL'])assert.equal(context.circuitRegion(league),'APAC');
 assert.equal(context.circuitRegion('CBLOL'),'BR');
 assert.equal(context.circuitRegion('LLA'),'LATAM');
 assert.equal(context.circuitRegion('LTA'),'AMERICAS');
});
test('international affiliation uses dated domestic history without future transfers',()=>{
 const player={history:[{date:'2022-12-31',league:'LCK'},{date:'2023-12-31',league:'LPL'}]};
 assert.equal(context.regionFor(player,'WLDs','2023-10-01'),'KR');
 assert.equal(context.regionFor(player,'WLDs','2024-10-01'),'CN');
 assert.equal(context.regionFor(player,'LPL','2023-10-01'),'CN');
});
test('unresolved events and unrecognized circuits remain explicit',()=>{
 assert.equal(context.regionFor({history:[]},'WSCI','2026-10-04'),'INT');
 assert.equal(context.regionFor({history:[{date:'2025-12-31',league:'LCK'}]},'new circuit','2026-10-04'),'UNKNOWN');
});
