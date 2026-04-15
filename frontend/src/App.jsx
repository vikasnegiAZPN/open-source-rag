import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [kbLoaded, setKbLoaded] = useState(false);

  const API_URL = 'http://localhost:8000';

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      alert('Please select a file');
      return;
    }

    setLoading(true);
    setUploadStatus('Uploading...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadStatus(`✅ ${response.data.message}`);
      setKbLoaded(true);
      setFile(null);
    } catch (error) {
      setUploadStatus(`❌ Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!question.trim()) {
      alert('Please enter a question');
      return;
    }

    if (!kbLoaded) {
      alert('Please upload a knowledge base first');
      return;
    }

    setLoading(true);
    setAnswer('');
    setSources([]);

    try {
      const response = await axios.post(`${API_URL}/query`, {
        question: question,
        top_k: 5
      });

      setAnswer(response.data.answer);
      setSources(response.data.sources);
    } catch (error) {
      setAnswer(`❌ Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      if (response.data.kb_loaded) {
        setKbLoaded(true);
      }
    } catch (error) {
      console.log('API not running');
    }
  };

  React.useEffect(() => {
    checkHealth();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 Open-Source RAG</h1>
        <p>Free Knowledge Base Q&A powered by Ollama</p>
      </header>

      <div className="container">
        {/* Upload Section */}
        <section className="section">
          <h2>📚 Upload Knowledge Base</h2>
          <form onSubmit={handleUpload} className="upload-form">
            <input
              type="file"
              accept=".xlsx,.xls"
              onChange={handleFileChange}
              disabled={loading}
              placeholder="Select Excel file"
            />
            <button type="submit" disabled={loading || !file}>
              {loading ? '⏳ Uploading...' : '📤 Upload'}
            </button>
          </form>
          {uploadStatus && <p className="status">{uploadStatus}</p>}
          {kbLoaded && <p className="success">✅ Knowledge Base Ready!</p>}
        </section>

        {/* Query Section */}
        {kbLoaded && (
          <section className="section">
            <h2>❓ Ask a Question</h2>
            <form onSubmit={handleQuery} className="query-form">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask anything about your knowledge base..."
                disabled={loading}
              />
              <button type="submit" disabled={loading}>
                {loading ? '⏳ Thinking...' : '🚀 Ask'}
              </button>
            </form>

            {answer && (
              <div className="result">
                <h3>💡 Answer:</h3>
                <p className="answer-text">{answer}</p>

                {sources.length > 0 && (
                  <div className="sources">
                    <h4>📖 Sources:</h4>
                    {sources.map((source, idx) => (
                      <div key={idx} className="source-item">
                        <p><strong>Source:</strong> {source.source}</p>
                        <p><strong>Row:</strong> {source.row}</p>
                        <p><strong>Content:</strong> {source.content}...</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </section>
        )}
      </div>

      <footer className="App-footer">
        <p>💰 Cost: $0 | 🚀 Powered by Ollama, LangChain & Chroma</p>
      </footer>
    </div>
  );
}

export default App;
