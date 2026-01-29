import { useState } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './App.css'

function App() {
  const [config, setConfig] = useState({
    initial_cash: 100000,
    fast_period: 10,
    slow_period: 30,
    start_date: '',
    end_date: '',
  })
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [useSynthetic, setUseSynthetic] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      if (useSynthetic) {
        // Synthetic data endpoint
        const response = await axios.post('/api/backtest/synthetic', config, {
          headers: { 'Content-Type': 'application/json' },
        })
        setResults(response.data)
      } else {
        // File upload endpoint
        if (!file) {
          setError('Please select a file')
          setLoading(false)
          return
        }

        const formData = new FormData()
        formData.append('file', file)
        
        // Add config as form data
        Object.keys(config).forEach(key => {
          if (config[key] !== '') {
            formData.append(key, config[key])
          }
        })

        const response = await axios.post('/api/backtest', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
        setResults(response.data)
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value)
  }

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🚀 Trading Bot Backtest</h1>
        <p>Test your trading strategies with realistic backtesting</p>
      </header>

      <div className="container">
        <div className="form-section">
          <form onSubmit={handleSubmit} className="form">
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={useSynthetic}
                  onChange={(e) => setUseSynthetic(e.target.checked)}
                />
                Use Synthetic Data (for testing)
              </label>
            </div>

            {!useSynthetic && (
              <div className="form-group">
                <label>Upload Data File (CSV or Parquet)</label>
                <input
                  type="file"
                  accept=".csv,.parquet"
                  onChange={(e) => setFile(e.target.files[0])}
                  required={!useSynthetic}
                />
              </div>
            )}

            <div className="form-row">
              <div className="form-group">
                <label>Initial Cash ($)</label>
                <input
                  type="number"
                  value={config.initial_cash}
                  onChange={(e) => setConfig({ ...config, initial_cash: parseFloat(e.target.value) })}
                  min="1000"
                  step="1000"
                />
              </div>

              <div className="form-group">
                <label>Fast SMA Period</label>
                <input
                  type="number"
                  value={config.fast_period}
                  onChange={(e) => setConfig({ ...config, fast_period: parseInt(e.target.value) })}
                  min="1"
                  max="100"
                />
              </div>

              <div className="form-group">
                <label>Slow SMA Period</label>
                <input
                  type="number"
                  value={config.slow_period}
                  onChange={(e) => setConfig({ ...config, slow_period: parseInt(e.target.value) })}
                  min="1"
                  max="200"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Start Date (optional)</label>
                <input
                  type="date"
                  value={config.start_date}
                  onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                />
              </div>

              <div className="form-group">
                <label>End Date (optional)</label>
                <input
                  type="date"
                  value={config.end_date}
                  onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                />
              </div>
            </div>

            <button type="submit" disabled={loading} className="submit-btn">
              {loading ? 'Running Backtest...' : 'Run Backtest'}
            </button>
          </form>
        </div>

        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {results && (
          <div className="results">
            <h2>Backtest Results</h2>

            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-label">Total Return</div>
                <div className="metric-value">{formatPercent(results.metrics.total_return)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">CAGR</div>
                <div className="metric-value">{formatPercent(results.metrics.cagr)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Sharpe Ratio</div>
                <div className="metric-value">{results.metrics.sharpe.toFixed(2)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Sortino Ratio</div>
                <div className="metric-value">{results.metrics.sortino.toFixed(2)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Max Drawdown</div>
                <div className="metric-value negative">{formatPercent(results.metrics.max_drawdown)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Win Rate</div>
                <div className="metric-value">{formatPercent(results.metrics.win_rate)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Number of Trades</div>
                <div className="metric-value">{results.metrics.num_trades}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Turnover</div>
                <div className="metric-value">{results.metrics.turnover.toFixed(2)}</div>
              </div>
            </div>

            {results.equity_curve && results.equity_curve.length > 0 && (
              <div className="chart-section">
                <h3>Equity Curve</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={results.equity_curve}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="timestamp"
                      tickFormatter={(value) => new Date(value).toLocaleDateString()}
                    />
                    <YAxis tickFormatter={(value) => formatCurrency(value)} />
                    <Tooltip
                      formatter={(value) => formatCurrency(value)}
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="equity"
                      stroke="#2563eb"
                      strokeWidth={2}
                      name="Portfolio Equity"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {results.trades && results.trades.length > 0 && (
              <div className="trades-section">
                <h3>Trades ({results.trades.length})</h3>
                <div className="trades-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>Cost</th>
                        <th>PnL</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.trades.slice(0, 50).map((trade, idx) => (
                        <tr key={idx}>
                          <td>{new Date(trade.timestamp).toLocaleString()}</td>
                          <td>{trade.symbol}</td>
                          <td className={trade.side.toLowerCase()}>{trade.side}</td>
                          <td>{parseFloat(trade.quantity).toFixed(2)}</td>
                          <td>{formatCurrency(parseFloat(trade.price))}</td>
                          <td>{formatCurrency(parseFloat(trade.cost))}</td>
                          <td className={parseFloat(trade.pnl) >= 0 ? 'positive' : 'negative'}>
                            {formatCurrency(parseFloat(trade.pnl))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {results.trades.length > 50 && (
                    <p className="trades-note">Showing first 50 of {results.trades.length} trades</p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
