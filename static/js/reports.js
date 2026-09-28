(() => {
  document.getElementById('printReport')?.addEventListener('click', () => window.print());
  const node = document.getElementById('reportData');
  if (!node || typeof Chart === 'undefined') return;
  const data = JSON.parse(node.textContent);
  new Chart(document.getElementById('eventTrendsChart'), {
    type:'line', data:{labels:data.days, datasets:[['Total Events','total','#2f6fed'],['High / Critical','high','#ef4444'],['Medium','medium','#f5a524'],['Low / Informational','low','#16a34a']].map(([label,key,color]) => ({label,data:data[key],borderColor:color,backgroundColor:color+'15',fill:key==='total',tension:.3,pointRadius:2,borderWidth:2}))},
    options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},interaction:{mode:'index',intersect:false},scales:{x:{grid:{display:false},ticks:{maxTicksLimit:7}},y:{beginAtZero:true,ticks:{precision:0}}}}
  });
  for(const [id,rows] of [['categoryDonut',data.categories],['qualityDonut',data.quality]]) {
    new Chart(document.getElementById(id),{type:'doughnut',data:{labels:rows.length?rows.map(x=>x.label):['No events'],datasets:[{data:rows.length?rows.map(x=>x.count):[1],backgroundColor:rows.length?rows.map(x=>x.color):['#e4ebf1'],borderColor:'#fff',borderWidth:3}]},options:{responsive:true,maintainAspectRatio:false,animation:false,cutout:'72%',plugins:{legend:{display:false},tooltip:{enabled:!!rows.length}}}});
  }
})();
