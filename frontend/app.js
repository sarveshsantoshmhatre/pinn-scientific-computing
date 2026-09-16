const root = document.documentElement;
const body = document.body;
const themeToggle = document.getElementById('themeToggle');

if (localStorage.getItem('pinn-theme') === 'dark') body.classList.add('dark');
themeToggle.addEventListener('click', () => {
  body.classList.toggle('dark');
  localStorage.setItem('pinn-theme', body.classList.contains('dark') ? 'dark' : 'light');
  drawAll();
});

const lossCanvas = document.getElementById('lossChart');
const heatCanvas = document.getElementById('heatmap');
const comparisonCanvas = document.getElementById('comparisonChart');
const slider = document.getElementById('timeSlider');
const timeValue = document.getElementById('timeValue');

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

function palette() {
  const dark = body.classList.contains('dark');
  return { text: dark ? '#dce8fb' : '#1a2b45', grid: dark ? '#2a3b55' : '#e5ebf3', blue: '#2d5bd7', orange: '#ee774c', green: '#2aa66f' };
}

function axes(ctx, w, h, pad) {
  const p = palette();
  ctx.strokeStyle = p.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.t + (h - pad.t - pad.b) * i / 5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }
  ctx.strokeStyle = p.text; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
}

function drawLoss() {
  const {ctx,w,h}=setupCanvas(lossCanvas), p=palette(), pad={l:48,r:18,t:18,b:35};
  axes(ctx,w,h,pad);
  const values = Array.from({length:70},(_,i)=>0.42*Math.exp(-i/10)+0.000015+0.000025*Math.abs(Math.sin(i*.8)));
  const max=Math.max(...values), min=0.00001;
  ctx.strokeStyle=p.blue; ctx.lineWidth=2.5; ctx.beginPath();
  values.forEach((v,i)=>{const x=pad.l+(w-pad.l-pad.r)*i/(values.length-1);const y=pad.t+(h-pad.t-pad.b)*(1-(Math.log10(v)-Math.log10(min))/(Math.log10(max)-Math.log10(min)));i?ctx.lineTo(x,y):ctx.moveTo(x,y)}); ctx.stroke();
  ctx.fillStyle=p.text;ctx.font='11px system-ui';ctx.fillText('Loss (log scale)',8,15);ctx.fillText('Epoch',w/2-18,h-7);ctx.fillText('0',pad.l-4,h-12);ctx.fillText('3,000',w-pad.r-30,h-12);
}

function heatColor(v) {
  // Smooth blue -> cyan -> yellow -> orange -> red palette for the heat visualization.
  const stops=[[0,[37,116,190]],[.25,[34,174,207]],[.5,[111,211,171]],[.72,[255,221,93]],[.88,[255,137,61]],[1,[230,55,43]]];
  let a=stops[0],b=stops[stops.length-1];
  for(let i=0;i<stops.length-1;i++){if(v>=stops[i][0]&&v<=stops[i+1][0]){a=stops[i];b=stops[i+1];break}}
  const q=(v-a[0])/(b[0]-a[0]); const c=a[1].map((x,i)=>Math.round(x+(b[1][i]-x)*q)); return `rgb(${c.join(',')})`;
}

function drawHeat() {
  const {ctx,w,h}=setupCanvas(heatCanvas); const t=Number(slider.value)/100; const cols=42,rows=25; const cw=w/cols,ch=h/rows;
  for(let r=0;r<rows;r++) for(let c=0;c<cols;c++){
    const x=c/(cols-1); const value=Math.sin(Math.PI*x)*Math.exp(-0.01*Math.PI*Math.PI*(t*100));
    ctx.fillStyle=heatColor(Math.max(0,Math.min(1,value))); ctx.fillRect(c*cw,r*ch,cw+1,ch+1);
  }
  ctx.fillStyle='rgba(8,20,40,.72)';ctx.fillRect(8,8,120,24);ctx.fillStyle='#fff';ctx.font='11px system-ui';ctx.fillText(`time = ${t.toFixed(2)}`,18,24);
  timeValue.textContent=`t = ${t.toFixed(2)}`;
}

function curve(ctx,w,h,color,fn) {
  const pad={l:38,r:15,t:20,b:30}; ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.beginPath();
  for(let i=0;i<=80;i++){const x=i/80;const y=Math.max(0,Math.min(1,fn(x)));const px=pad.l+(w-pad.l-pad.r)*x;const py=pad.t+(h-pad.t-pad.b)*(1-y);i?ctx.lineTo(px,py):ctx.moveTo(px,py)}ctx.stroke();
}
function drawComparison(){
  const {ctx,w,h}=setupCanvas(comparisonCanvas),p=palette(); axes(ctx,w,h,{l:38,r:15,t:20,b:30});
  curve(ctx,w,h,p.green,x=>.88*Math.exp(-2.1*x));
  curve(ctx,w,h,p.blue,x=>.88*Math.exp(-2.08*x)+.018*Math.sin(10*x)*Math.exp(-2*x));
  curve(ctx,w,h,p.orange,x=>.88*Math.exp(-2.02*x)+.035*Math.sin(7*x)*Math.exp(-1.4*x));
  ctx.fillStyle=p.text;ctx.font='10px system-ui';ctx.fillText('Temperature',5,14);ctx.fillText('Position x',w/2-20,h-7);
}
function drawAll(){drawLoss();drawHeat();drawComparison()}
slider.addEventListener('input',drawHeat);
window.addEventListener('resize',drawAll);
requestAnimationFrame(drawAll);
