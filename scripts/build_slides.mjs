// Optional editable PowerPoint export. Requires the Codex Artifact Tool runtime.
// Run after python scripts/build_deliverables.py. Set ARTIFACT_TOOL_ENTRY to
// the absolute artifact_tool.mjs path when the package is not on Node's search path.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const { Presentation, PresentationFile } = await import(process.env.ARTIFACT_TOOL_ENTRY ? pathToFileURL(process.env.ARTIFACT_TOOL_ENTRY).href : '@oai/artifact-tool');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const tmp=path.join(root,'tmp/slides');
await fs.mkdir(tmp,{recursive:true});
const content=JSON.parse(await fs.readFile(path.join(root,'report/demo_content.json'),'utf8'));
function csv(text){const lines=text.trim().split(/\r?\n/);const fields=lines.shift().split(',');return lines.map(line=>Object.fromEntries(line.split(',').map((v,i)=>[fields[i],v])));}
const summary=csv(await fs.readFile(path.join(root,'results/summary.csv'),'utf8'));
const metrics=csv(await fs.readFile(path.join(root,'results/metrics.csv'),'utf8'));
const palette=['#8795a6','#2879c0','#e49428','#159b85'];
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
function textbox(slide,text,x,y,w,h,size=30,color='#20334d',bold=false){
 const box=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 box.text=text;box.text.style={typeface:'Arial',fontSize:size,color,bold};return box;
}
for(const [index,item] of content.entries()){
 const slide=presentation.slides.add();slide.background.fill=index===0?'#12334b':'#f7f9fb';
 slide.speakerNotes.text=item.notes+' Source: local project results/metrics.csv and supplied EC417 P22 brief.';
 textbox(slide,item.title,72,index===0?145:52,1136,100,index===0?56:44,index===0?'#ffffff':'#12334b',true);
 if(item.kind==='results'){
  const values=[['Model','MAE (C)','RMSE (C)','Time (s)'],...summary.map(r=>[r.model,`${(+r.mae_mean).toFixed(3)} +/- ${(+r.mae_std).toFixed(3)}`,`${(+r.rmse_mean).toFixed(3)} +/- ${(+r.rmse_std).toFixed(3)}`,(+r.train_seconds_mean).toFixed(1)])];
  const table=slide.tables.add({rows:5,columns:4,left:72,top:190,width:1136,height:300,columnWidths:[300,320,320,196],values});
  table.cells.block({row:0,column:0,rowCount:5,columnCount:4}).assign({textStyle:{typeface:'Arial',fontSize:25,color:'#20334d'},fill:'#eef4f6'});
  table.cells.block({row:0,column:0,rowCount:1,columnCount:4}).assign({textStyle:{typeface:'Arial',fontSize:25,bold:true,color:'#ffffff'},fill:'#12334b'});
  textbox(slide,item.body,72,530,1136,100,27);
  textbox(slide,'Mean +/- sample SD across 3 seeds. Deterministic baselines run once.',72,652,1136,35,20);
 }else if(item.kind==='horizon'){
  const horizons=[1,6,12,24];
  const series=summary.map((r,i)=>({name:r.model,values:horizons.map(h=>{const a=metrics.filter(m=>m.model===r.model&&m.metric==='MAE'&&+m.horizon_hours===h);return +(a.reduce((s,m)=>s+(+m.value),0)/a.length).toFixed(6);}),line:{fill:palette[i],width:3},marker:{symbol:'circle',size:7,fill:palette[i]}}));
  const chart=slide.charts.add('line',{position:{left:90,top:170,width:1100,height:450},categories:horizons.map(h=>`${h}h`),series,hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:22}},xAxis:{title:'Forecast horizon',textStyle:{fontSize:22}},yAxis:{title:'Test MAE (degrees C)',textStyle:{fontSize:22},numberFormatCode:'0.0',majorGridlines:{fill:'#dbe3eb',width:1}},chartFill:'#f7f9fb',plotAreaFill:'#f7f9fb'});
  for(const style of [chart.legend.textStyle,chart.xAxis.textStyle,chart.yAxis.textStyle,chart.xAxis.title.textStyle,chart.yAxis.title.textStyle]){if(style)style.typeface='Arial';}
  textbox(slide,'Mean across seeds. The report shows seed variability with error bars.',90,645,1100,40,22);
 }else{
  const lines=item.body.split('\n');
  lines.forEach((line,j)=>textbox(slide,line,72,(index===0?325:190)+j*76,1136,65,index===0?31:31,index===0?'#d8e9ed':'#20334d'));
 }
 textbox(slide,String(index+1).padStart(2,'0'),1180,675,50,30,18,index===0?'#d8e9ed':'#65768a');
}
await (await PresentationFile.exportPptx(presentation)).save(path.join(tmp,'candidate.pptx'));
for(const [i,slide] of presentation.slides.items.entries()){
 const png=await presentation.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log('Exported editable draft and slide previews to tmp/slides.');

