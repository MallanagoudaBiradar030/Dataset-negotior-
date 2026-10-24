import { useState } from 'react';

type Dataset = {id:string;name:string;domain:string;rows:number;features:number;quality:number;diversity:number;redundancy:number;compute:string;description:string;score:number;score_breakdown:Record<string,number>};
const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const examples = ['Image classification for a small computer vision project','Predict housing prices using tabular data','Sentiment analysis on customer reviews'];
export default function App() {
 const [task,setTask] = useState(examples[0]);
 const [size,setSize] = useState('any');
 const [compute,setCompute] = useState('medium');
 const [results,setResults] = useState<Dataset[]>([]);
 const [loading,setLoading] = useState(false);
 const [error,setError] = useState('');
 async function search(e:React.FormEvent) {
  e.preventDefault(); setLoading(true); setError('');
  try {
   const response = await fetch(API + '/api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task,rows_preference:size,compute_budget:compute,limit:5})});
   if(!response.ok) throw new Error('API returned ' + response.status);
   const data = await response.json(); setResults(data.recommendations);
  } catch { setError('Could not reach the API. Set VITE_API_URL to your deployed backend URL, or start the backend locally on port 8000.'); }
  finally { setLoading(false); }
 }
 return <main className="shell">
  <nav><div className="brand-mark">DN</div><span>DATASET NEGOTIATOR</span><span className="nav-tag">BETA · EXPLAINABLE AI</span></nav>
  <section className="hero"><p className="eyebrow">YOUR NEXT ML PROJECT STARTS HERE</p><h1>Find the <em>right data.</em><br/>Build something meaningful.</h1><p className="subtitle">Describe your machine learning goal. We'll rank datasets using transparent, weighted criteria—not black-box guesses.</p></section>
  <section className="panel"><div className="panel-heading"><div><p className="eyebrow">01 / DEFINE YOUR TASK</p><h2>What are you building?</h2></div><span className="step">INPUT</span></div>
   <form onSubmit={search}><label htmlFor="task">Machine learning objective</label><textarea id="task" value={task} onChange={e=>setTask(e.target.value)} minLength={2} maxLength={500} required placeholder="e.g. Detect defects in manufacturing images..." />
   <div className="examples">{examples.map(x=><button type="button" key={x} onClick={()=>setTask(x)}>{x}</button>)}</div>
   <div className="filters"><div><label htmlFor="size">Dataset size</label><select id="size" value={size} onChange={e=>setSize(e.target.value)}><option value="any">Any size</option><option value="small">Small (&lt; 10k rows)</option><option value="large">Large (10k+ rows)</option></select></div><div><label htmlFor="compute">Compute budget</label><select id="compute" value={compute} onChange={e=>setCompute(e.target.value)}><option value="low">Low — CPU / laptop</option><option value="medium">Medium — capable laptop</option><option value="high">High — GPU / cloud</option></select></div></div>
   <button className="primary" disabled={loading}>{loading?'Ranking datasets…':'Recommend datasets'} <span>↗</span></button></form>
  </section>
  {error && <div className="error">{error}</div>}
  {results.length>0 && <section className="results"><div className="panel-heading"><div><p className="eyebrow">02 / RECOMMENDATIONS</p><h2>Your best-fit datasets</h2></div><span className="step">{results.length} MATCHES</span></div>{results.map((d,i)=><article className="dataset" key={d.id}><div className="rank">0{i+1}</div><div className="dataset-main"><div className="dataset-title"><h3>{d.name}</h3><span>{d.compute.toUpperCase()} COMPUTE</span></div><p>{d.description}</p><div className="meta"><span>{d.rows.toLocaleString()} rows</span><span>{d.features.toLocaleString()} features</span><span>{d.domain}</span></div><details><summary>Why this score?</summary><div className="breakdown">{Object.entries(d.score_breakdown).map(([k,v])=><div key={k}><span>{k.replaceAll('_',' ')}</span><div className="track"><i style={{width:v+'%'}} /></div><b>{v}%</b></div>)}</div></details></div><div className="score"><strong>{d.score}</strong><span>FIT SCORE</span></div></article>)}</section>}
  <section className="method"><p className="eyebrow">HOW RANKING WORKS</p><h2>Transparent by design.</h2><p>Every recommendation uses the same weighted scoring model. Inspect the breakdown to understand how task relevance, quality, diversity, redundancy, size, and compute fit affect each result.</p><div className="weights">{[['Relevance','35%'],['Data quality','20%'],['Diversity','15%'],['Low redundancy','10%'],['Size fit','10%'],['Compute fit','10%']].map(([label,value])=><div key={label}><span>{label}</span><b>{value}</b></div>)}</div></section>
  <footer><span>AI DATASET NEGOTIATOR</span><span>Built for practical, responsible ML discovery.</span></footer>
 </main>;
}
