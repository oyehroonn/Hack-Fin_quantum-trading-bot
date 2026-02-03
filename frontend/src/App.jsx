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
  
  // Helper function to safely parse numeric values
  const parseNumeric = (value) => {
    if (value === '' || value === null || value === undefined) return 0
    const parsed = parseFloat(value)
    return isNaN(parsed) ? 0 : parsed
  }
  
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [useSynthetic, setUseSynthetic] = useState(false)
  const [useRealData, setUseRealData] = useState(false)
  const [useCrypto, setUseCrypto] = useState(false)
  const [symbol, setSymbol] = useState('AAPL')
  const [symbolSearch, setSymbolSearch] = useState('')
  const [symbolOptions, setSymbolOptions] = useState([])
  const [periodDays, setPeriodDays] = useState(30)
  const [assetClass, setAssetClass] = useState('equities')
  const [suggestions, setSuggestions] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      if (useCrypto) {
        // Crypto data endpoint (Binance)
        const params = new URLSearchParams({
          symbol: symbol.toUpperCase().replace('/', ''),
          initial_cash: config.initial_cash.toString(),
          fast_period: config.fast_period.toString(),
          slow_period: config.slow_period.toString(),
          timeframe: '1d',
        })
        
        if (config.start_date) params.append('start_date', config.start_date)
        if (config.end_date) params.append('end_date', config.end_date)
        
        const response = await axios.post(
          `/api/backtest/crypto?${params.toString()}`,
          {},
          {
            headers: { 'Content-Type': 'application/json' },
          }
        )
        setResults(response.data)
      } else if (useRealData) {
        // Real stock data endpoint
        const params = new URLSearchParams({
          symbol: symbol.toUpperCase(),
          initial_cash: config.initial_cash.toString(),
          fast_period: config.fast_period.toString(),
          slow_period: config.slow_period.toString(),
          timeframe: '1d',
        })
        
        if (config.start_date) params.append('start_date', config.start_date)
        if (config.end_date) params.append('end_date', config.end_date)
        
        const response = await axios.post(
          `/api/backtest/real?${params.toString()}`,
          {},
          {
            headers: { 'Content-Type': 'application/json' },
          }
        )
        setResults(response.data)
      } else if (useSynthetic) {
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
              <label>Asset Class</label>
              <select
                value={assetClass}
                onChange={(e) => {
                  setAssetClass(e.target.value)
                  setUseCrypto(e.target.value === 'crypto')
                  setUseRealData(e.target.value === 'equities')
                  setUseSynthetic(false)
                  if (e.target.value === 'crypto') {
                    setSymbol('BTCUSDT')
                  } else {
                    setSymbol('AAPL')
                  }
                }}
                style={{ padding: '8px', fontSize: '1em', width: '100%' }}
              >
                <option value="equities">Stocks (Equities)</option>
                <option value="crypto">Cryptocurrency</option>
              </select>
            </div>

            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={useSynthetic}
                  onChange={(e) => {
                    setUseSynthetic(e.target.checked)
                    if (e.target.checked) {
                      setUseRealData(false)
                      setUseCrypto(false)
                    }
                  }}
                />
                Use Synthetic Data (for testing)
              </label>
            </div>

            {(useRealData || useCrypto) && (
              <>
                <div className="form-group">
                  <label>{assetClass === 'crypto' ? 'Crypto Symbol' : 'Stock Symbol'}</label>
                  <input
                    type="text"
                    value={symbol}
                    onChange={async (e) => {
                      const value = e.target.value.toUpperCase()
                      setSymbol(value)
                      setSymbolSearch(value)
                      
                      // Search symbols as user types
                      if (value.length >= 1) {
                        try {
                          const response = await axios.get(
                            `/api/symbols/search?query=${encodeURIComponent(value)}&asset_class=${assetClass}`
                          )
                          setSymbolOptions(response.data.symbols || [])
                        } catch (err) {
                          console.error('Symbol search error:', err)
                        }
                      } else {
                        setSymbolOptions([])
                      }
                    }}
                    placeholder={assetClass === 'crypto' ? 'BTCUSDT, ETHUSDT, etc.' : 'AAPL, MSFT, GOOGL, etc.'}
                    style={{ textTransform: 'uppercase' }}
                    list="symbol-list"
                  />
                  <datalist id="symbol-list">
                    {symbolOptions.map((opt, idx) => (
                      <option key={idx} value={opt.symbol}>{opt.display}</option>
                    ))}
                  </datalist>
                  <small style={{ color: '#666', fontSize: '0.9em', display: 'block', marginTop: '4px' }}>
                    {assetClass === 'crypto' 
                      ? 'Enter crypto pair (e.g., BTCUSDT, ETHUSDT) - Data from Binance'
                      : 'Enter stock symbol - Data from Yahoo Finance'}
                  </small>
                </div>

                <div className="form-group">
                  <label>Period (Days)</label>
                  <select
                    value={periodDays}
                    onChange={(e) => {
                      const days = parseInt(e.target.value)
                      setPeriodDays(days)
                      const end = new Date()
                      const start = new Date()
                      start.setDate(start.getDate() - days)
                      setConfig({
                        ...config,
                        start_date: start.toISOString().split('T')[0],
                        end_date: end.toISOString().split('T')[0],
                      })
                    }}
                    style={{ padding: '8px', fontSize: '1em', width: '100%' }}
                  >
                    <option value={7}>Last 7 days</option>
                    <option value={30}>Last 30 days</option>
                    <option value={90}>Last 90 days</option>
                    <option value={180}>Last 6 months</option>
                    <option value={365}>Last year</option>
                    <option value={730}>Last 2 years</option>
                  </select>
                </div>

                <div className="form-group">
                  <button
                    type="button"
                    onClick={async () => {
                      try {
                        setLoading(true)
                        const response = await axios.post('/api/backtest/suggest', {
                          symbol: symbol,
                          asset_class: assetClass,
                          initial_cash: config.initial_cash,
                          fast_period: config.fast_period,
                          slow_period: config.slow_period,
                          period_days: periodDays,
                          timeframe: '1d',
                        })
                        setSuggestions(response.data)
                      } catch (err) {
                        setError(err.response?.data?.detail || err.message)
                      } finally {
                        setLoading(false)
                      }
                    }}
                    style={{
                      padding: '10px 20px',
                      backgroundColor: '#8b5cf6',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '1em',
                      width: '100%',
                    }}
                  >
                    Get Trade Suggestion
                  </button>
                </div>

                {suggestions && (
                  <div style={{
                    padding: '15px',
                    backgroundColor: '#f3f4f6',
                    borderRadius: '8px',
                    marginTop: '10px',
                    border: `2px solid ${suggestions.signal === 'BUY' ? '#10b981' : suggestions.signal === 'SELL' ? '#ef4444' : '#6b7280'}`,
                  }}>
                    <h3 style={{ marginTop: 0, color: suggestions.signal === 'BUY' ? '#10b981' : suggestions.signal === 'SELL' ? '#ef4444' : '#6b7280' }}>
                      {suggestions.signal} {suggestions.symbol}
                    </h3>
                    <p><strong>Confidence:</strong> {(suggestions.confidence * 100).toFixed(0)}%</p>
                    <p><strong>Current Price:</strong> ${suggestions.current_price?.toFixed(2)}</p>
                    <p><strong>Risk Level:</strong> {suggestions.risk_level}</p>
                    <p><strong>Expected Monthly Return:</strong> {(suggestions.expected_monthly_return * 100).toFixed(2)}%</p>
                    <p><strong>Recommendation:</strong> {suggestions.recommendation}</p>
                  </div>
                )}
              </>
            )}

            {!useSynthetic && !useRealData && (
              <div className="form-group">
                <label>Upload Data File (CSV or Parquet)</label>
                <input
                  type="file"
                  accept=".csv,.parquet"
                  onChange={(e) => setFile(e.target.files[0])}
                  required={!useSynthetic && !useRealData}
                />
              </div>
            )}

            <div className="form-row">
              <div className="form-group">
                <label>Initial Cash ($)</label>
                <input
                  type="number"
                  value={config.initial_cash || ''}
                  onChange={(e) => {
                    const val = parseNumeric(e.target.value)
                    setConfig({ ...config, initial_cash: val || 100000 })
                  }}
                  min="1000"
                  step="1000"
                />
              </div>

              <div className="form-group">
                <label>Fast SMA Period</label>
                <input
                  type="number"
                  value={config.fast_period || ''}
                  onChange={(e) => {
                    const val = parseNumeric(e.target.value)
                    setConfig({ ...config, fast_period: val || 10 })
                  }}
                  min="1"
                  max="100"
                />
              </div>

              <div className="form-group">
                <label>Slow SMA Period</label>
                <input
                  type="number"
                  value={config.slow_period || ''}
                  onChange={(e) => {
                    const val = parseNumeric(e.target.value)
                    setConfig({ ...config, slow_period: val || 30 })
                  }}
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
