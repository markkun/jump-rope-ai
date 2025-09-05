import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [results, setResults] = useState([]);

  const uploadVideo = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('video', file);
    const res = await axios.post('http://localhost:5000/analyze', formData);
    setResults(res.data);
  };

  return (
    <div>
      <h1>🏃‍♂️ 跳绳 AI 分析系统</h1>
      <input type="file" accept="video/*" onChange={uploadVideo} />
      <div>
        {results.map(r => (
          <div key={r.id}>
            ID: {r.id} | 次数: {r.count} | 评分: {r.score.toFixed(1)}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
