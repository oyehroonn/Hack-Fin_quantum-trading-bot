import { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import CandlestickChart from './components/CandlestickChart'
import { useLivePrice } from './hooks/useLivePrice'
import { useBinanceWebSocket } from './hooks/useBinanceWebSocket'
import { useTerminalSession } from './hooks/useTerminalSession'
import BotDashboard from './components/BotDashboard'
import FuturesDashboard from './components/FuturesDashboard'
import './App.css'

function formatCurrency(value) {
  if (value == null || isNaN(value)) return '$0.00'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(value)
}

function formatPercent(value) {
  if (value == null || isNaN(value)) return '0.00%'
  return `${(value * 100).toFixed(2)}%`
}

function safeError(detail) {
  if (detail == null) return 'An error occurred'
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((e) => e.msg || JSON.stringify(e)).join('; ')
  return String(detail)
}

function App() {
  const [view, setView] = useState('terminal')
  const [config, setConfig] = useState({
    initial_cash: 100000,
    fast_period: 10,
    slow_period: 30,
    start_date: '',
    end_date: '',
  })
  const parseNumeric = (value) => {
    if (value === '' || value == null) return 0
    const parsed = parseFloat(value)
    return isNaN(parsed) ? 0 : parsed
  }

  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [useSynthetic, setUseSynthetic] = useState(false)
  const [useRealData, setUseRealData] = useState(true)
  const [useCrypto, setUseCrypto] = useState(false)
  const [symbol, setSymbol] = useState('AAPL')
  const [symbolSearch, setSymbolSearch] = useState('')
  const [symbolOptions, setSymbolOptions] = useState([])
  const [periodDays, setPeriodDays] = useState(30)
  const [assetClass, setAssetClass] = useState('equities')
  const [suggestions, setSuggestions] = useState(null)
  const [suggestLoading, setSuggestLoading] = useState(false)

  const [orderAmount, setOrderAmount] = useState(100)
  const [ohlcv, setOhlcv] = useState([])
  const [chartTimeframe, setChartTimeframe] = useState('1h')
  const [sessionIdInput, setSessionIdInput] = useState('')

  const { sessionId, portfolio, positions, trades, loading: terminalLoading, error: terminalError, clearError: clearTerminalError, executeTrade, refetch: refetchTerminal, restoreSession } = useTerminalSession()
  const { price: livePrice, loading: priceLoading, refresh: refreshPrice } = useLivePrice(symbol, assetClass, 5000)
  const { price: wsPrice, kline: wsKline, connected: wsConnected } = useBinanceWebSocket(symbol, chartTimeframe, assetClass === 'crypto' && view === 'terminal')

  const fetchSuggest = async () => {
    try {
      setSuggestLoading(true)
      setError(null)
      const params = new URLSearchParams({
        symbol: symbol.toUpperCase().replace('/', ''),
        asset_class: assetClass,
        initial_cash: String(config.initial_cash),
        fast_period: String(config.fast_period),
        slow_period: String(config.slow_period),
        period_days: String(periodDays),
        timeframe: '1d',
      })
      const url = `/api/backtest/suggest?${params}`
      const res = await axios.post(url, {})
      setSuggestions(res.data)
    } catch (err) {
      setError(safeError(err.response?.data?.detail || err.message))
      setSuggestions(null)
    } finally {
      setSuggestLoading(false)
    }
  }

  const currentPrice = (assetClass === 'crypto' ? wsPrice : null) ?? livePrice ?? suggestions?.current_price

  const sellPosition = async (pos) => {
    const sym = pos.symbol?.toUpperCase().replace('/', '')
    if (!sym || !pos?.qty || pos.qty <= 0) return
    let price = currentPrice
    if (sym !== symbol.toUpperCase().replace('/', '')) {
      try {
        const res = await axios.get('/api/market/price', { params: { symbol: sym, asset_class: assetClass } })
        price = res.data?.price
      } catch {
        setError('Could not fetch price for ' + sym)
        return
      }
    }
    if (!price) return
    await executeTrade({ symbol: sym, side: 'SELL', qty: pos.qty, price, asset_class: assetClass })
  }

  const executeAction = async (action) => {
    if (!currentPrice) return
    const sym = symbol.toUpperCase().replace('/', '')
    if (action === 'BUY') {
      const amountUsd = typeof orderAmount === 'number' ? orderAmount : parseFloat(String(orderAmount).replace(/[^0-9.-]/g, '')) || 0
      const cash = portfolio?.cash ?? 0
      if (amountUsd <= 0 || amountUsd > cash) return
      const qty = amountUsd / currentPrice
      const decimals = assetClass === 'crypto' ? 8 : 4
      const qtyRounded = parseFloat(qty.toFixed(decimals))
      if (qtyRounded <= 0) return
      await executeTrade({ symbol: sym, side: 'BUY', qty: qtyRounded, price: currentPrice, asset_class: assetClass })
    } else if (action === 'SELL') {
      const pos = positions.find((p) => p.symbol === sym)
      if (!pos?.qty || pos.qty <= 0) return
      await executeTrade({ symbol: sym, side: 'SELL', qty: pos.qty, price: currentPrice, asset_class: assetClass })
    }
  }

  const totalEquity = () => {
    const cash = portfolio?.cash ?? 0
    let eq = cash
    ;(positions || []).forEach((p) => {
      const mark = p.symbol === symbol.toUpperCase().replace('/', '') ? currentPrice : null
      eq += p.qty * (mark ?? p.avg_cost)
    })
    return eq
  }

  const realizedPnl = () => portfolio?.realized_pnl ?? 0

  useEffect(() => {
    if (!symbol?.trim() || view !== 'terminal') return
    const fetchOhlcv = async () => {
      try {
        const res = await axios.get('/api/market/ohlcv', {
          params: { symbol: symbol.toUpperCase().replace('/', ''), asset_class: assetClass, timeframe: chartTimeframe, days: 60 },
        })
        setOhlcv(res.data?.ohlcv ?? [])
      } catch {
        setOhlcv([])
      }
    }
    fetchOhlcv()
    const interval = chartTimeframe === '1d' ? 120000 : 60000
    const id = setInterval(fetchOhlcv, interval)
    return () => clearInterval(id)
  }, [symbol, assetClass, view, chartTimeframe])

  useEffect(() => {
    if (view === 'terminal' && (useRealData || useCrypto) && symbol) {
      fetchSuggest()
    }
  }, [view, symbol, assetClass, periodDays])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      if (useCrypto) {
        const params = new URLSearchParams({
          symbol: symbol.toUpperCase().replace('/', ''),
          initial_cash: config.initial_cash.toString(),
          fast_period: config.fast_period.toString(),
          slow_period: config.slow_period.toString(),
          timeframe: '1d',
        })
        if (config.start_date) params.append('start_date', config.start_date)
        if (config.end_date) params.append('end_date', config.end_date)
        const response = await axios.post(`/api/backtest/crypto?${params}`, {}, { headers: { 'Content-Type': 'application/json' } })
        setResults(response.data)
      } else if (useRealData) {
        const params = new URLSearchParams({
          symbol: symbol.toUpperCase(),
          initial_cash: config.initial_cash.toString(),
          fast_period: config.fast_period.toString(),
          slow_period: config.slow_period.toString(),
          timeframe: '1d',
        })
        if (config.start_date) params.append('start_date', config.start_date)
        if (config.end_date) params.append('end_date', config.end_date)
        const response = await axios.post(`/api/backtest/real?${params}`, {}, { headers: { 'Content-Type': 'application/json' } })
        setResults(response.data)
      } else if (useSynthetic) {
        const response = await axios.post('/api/backtest/synthetic', config, { headers: { 'Content-Type': 'application/json' } })
        setResults(response.data)
      } else {
        if (!file) {
          setError('Please select a file')
          setLoading(false)
          return
        }
        const formData = new FormData()
        formData.append('file', file)
        Object.keys(config).forEach((key) => {
          if (config[key] !== '') formData.append(key, config[key])
        })
        const response = await axios.post('/api/backtest', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
        setResults(response.data)
      }
    } catch (err) {
      setError(safeError(err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Quantum Trading Bot</h1>
        <p>Live terminal & backtest your strategies</p>
        <nav className="nav-tabs">
          <button className={view === 'terminal' ? 'active' : ''} onClick={() => setView('terminal')}>
            Trading Terminal
          </button>
          <button className={view === 'bot' ? 'active' : ''} onClick={() => setView('bot')}>
            Spot Bot
          </button>
          <button className={view === 'futures' ? 'active' : ''} onClick={() => setView('futures')}>
            Futures Bot
          </button>
          <button className={view === 'backtest' ? 'active' : ''} onClick={() => setView('backtest')}>
            Backtest
          </button>
        </nav>
      </header>

      <div className="container">
        {(error || terminalError) && (
          <div className="error">
            <strong>Error:</strong> {error || terminalError}
            <button type="button" className="dismiss" onClick={() => { setError(null); clearTerminalError() }}>×</button>
          </div>
        )}

        {view === 'terminal' && (
          <div className="terminal">
            <div className="terminal-sidebar">
              <div className="terminal-form">
                <label>Symbol</label>
                <input
                  type="text"
                  value={symbol}
                  onChange={async (e) => {
                    const v = e.target.value.toUpperCase()
                    setSymbol(v)
                    if (v.length >= 1) {
                      try {
                        const res = await axios.get(`/api/symbols/search?query=${encodeURIComponent(v)}&asset_class=${assetClass}`)
                        setSymbolOptions(res.data.symbols || [])
                      } catch {
                        setSymbolOptions([])
                      }
                    } else setSymbolOptions([])
                  }}
                  list="symbol-list"
                  placeholder="AAPL, BTCUSDT..."
                />
                <datalist id="symbol-list">
                  {symbolOptions.map((o, i) => (
                    <option key={i} value={o.symbol}>{o.display}</option>
                  ))}
                </datalist>
                <label>Asset Class</label>
                <select
                  value={assetClass}
                  onChange={(e) => {
                    setAssetClass(e.target.value)
                    setUseCrypto(e.target.value === 'crypto')
                    setUseRealData(e.target.value === 'equities')
                    setSymbol(e.target.value === 'crypto' ? 'BTCUSDT' : 'AAPL')
                  }}
                >
                  <option value="equities">Stocks</option>
                  <option value="crypto">Crypto</option>
                </select>
                <label>Period (days)</label>
                <select value={periodDays} onChange={(e) => setPeriodDays(Number(e.target.value))}>
                  <option value={7}>7</option>
                  <option value={30}>30</option>
                  <option value={90}>90</option>
                  <option value={365}>365</option>
                </select>
                <button type="button" className="btn-refresh" onClick={fetchSuggest} disabled={suggestLoading}>
                  {suggestLoading ? 'Loading...' : 'Refresh'}
                </button>
              </div>
              <div className="session-section">
                <h3>Session</h3>
                <div className="session-id-row">
                  <code className="session-id">{sessionId ? sessionId.slice(0, 8) + '…' : '—'}</code>
                  <button type="button" className="btn-copy" onClick={() => { if (sessionId) navigator.clipboard.writeText(sessionId) }} title="Copy full Session ID">
                    Copy
                  </button>
                </div>
                <div className="restore-session">
                  <input
                    type="text"
                    placeholder="Restore session ID"
                    value={sessionIdInput}
                    onChange={(e) => setSessionIdInput(e.target.value)}
                    className="session-id-input"
                  />
                  <button type="button" className="btn-restore" onClick={() => restoreSession(sessionIdInput)} disabled={terminalLoading || !sessionIdInput?.trim()}>
                    Restore
                  </button>
                </div>
              </div>
              <div className="portfolio-summary">
                <h3>Portfolio</h3>
                <div className="stat"><span>Cash</span><span>{formatCurrency(portfolio?.cash ?? 0)}</span></div>
                <div className="stat"><span>Equity</span><span>{formatCurrency(totalEquity())}</span></div>
                <div className="stat"><span>Realized P&L</span><span className={realizedPnl() >= 0 ? 'positive' : 'negative'}>{formatCurrency(realizedPnl())}</span></div>
              </div>
              <div className="positions-list">
                <h3>Positions</h3>
                {!positions?.length ? (
                  <p className="muted">No positions</p>
                ) : (
                  positions.map((p) => {
                    if (!p?.qty) return null
                    const sym = p.symbol
                    const mkt = sym === symbol.toUpperCase().replace('/', '') ? currentPrice : p.avg_cost
                    const unrealized = (mkt - p.avg_cost) * p.qty
                    return (
                      <div key={sym} className="position-row">
                        <div className="position-info">
                          <span>{sym}</span>
                          <span>{(typeof p.qty === 'number' && p.qty < 1 ? p.qty.toFixed(8).replace(/\.?0+$/, '') : p.qty)} @ {formatCurrency(p.avg_cost)}</span>
                          <span className={unrealized >= 0 ? 'positive' : 'negative'}>{formatCurrency(unrealized)}</span>
                        </div>
                        <button type="button" className="btn-sell-small" onClick={() => sellPosition(p)} disabled={terminalLoading}>Sell</button>
                      </div>
                    )
                  })
                )}
              </div>
            </div>
            <div className="terminal-main">
              <div className="chart-area">
                <div className="chart-header">
                  <select value={chartTimeframe} onChange={(e) => setChartTimeframe(e.target.value)} className="chart-timeframe-select">
                    <option value="1m">1m</option>
                    <option value="5m">5m</option>
                    <option value="15m">15m</option>
                    <option value="1h">1h</option>
                    <option value="1d">1D</option>
                  </select>
                </div>
                <CandlestickChart ohlcv={ohlcv} symbol={symbol} height={360} livePrice={assetClass === 'crypto' ? wsPrice ?? currentPrice : currentPrice} liveKline={assetClass === 'crypto' ? wsKline : null} />
              </div>
              <div className="market-card">
                <h2>{symbol}</h2>
                <div className="price">{formatCurrency(currentPrice)}{assetClass === 'crypto' && wsConnected ? <span className="price-live"> ● Live</span> : priceLoading ? <span className="price-loading"> (updating…)</span> : null}</div>
                <div className="signal-row">
                  <span className={`badge signal-${(suggestions?.signal || 'HOLD').toLowerCase()}`}>
                    {suggestions?.signal || 'HOLD'}
                  </span>
                  <span>Confidence: {suggestions?.confidence != null ? `${(suggestions.confidence * 100).toFixed(0)}%` : '—'}</span>
                  <span>Risk: {suggestions?.risk_level || '—'}</span>
                  {suggestions?.regime && <span>Regime: {suggestions.regime}</span>}
                </div>
                {suggestions?.recommendation && (
                  <p className="recommendation">{suggestions.recommendation}</p>
                )}
                <div className="order-amount-section">
                  <label>Order amount ($)</label>
                  <div className="order-amount-buttons">
                    {[100, 500, 1000].map((amt) => (
                      <button key={amt} type="button" className={`order-amt-btn ${orderAmount === amt ? 'active' : ''}`} onClick={() => setOrderAmount(amt)}>
                        ${amt}
                      </button>
                    ))}
                    <button type="button" className={`order-amt-btn ${orderAmount === Math.floor((portfolio?.cash ?? 0) * 0.1) ? 'active' : ''}`} onClick={() => setOrderAmount(Math.floor((portfolio?.cash ?? 0) * 0.1))}>
                      10%
                    </button>
                  </div>
                  <input
                    type="number"
                    className="order-amount-input"
                    value={orderAmount}
                    onChange={(e) => setOrderAmount(Math.max(0, parseFloat(e.target.value) || 0))}
                    min="1"
                    step="1"
                    placeholder="Custom amount"
                  />
                </div>
                <div className="action-buttons">
                  <button type="button" className="btn-buy" onClick={() => executeAction('BUY')} disabled={!currentPrice || suggestLoading || terminalLoading || orderAmount <= 0 || orderAmount > (portfolio?.cash ?? 0)}>
                    BUY ${orderAmount}
                  </button>
                  <button type="button" className="btn-sell" onClick={() => executeAction('SELL')} disabled={!positions?.find((p) => p.symbol === symbol.toUpperCase().replace('/', ''))?.qty || suggestLoading || terminalLoading}>
                    SELL
                  </button>
                  <button type="button" className="btn-hold" disabled>HOLD</button>
                </div>
              </div>
              <div className="trades-history">
                <div className="trades-history-header">
                  <h3>Recent Trades</h3>
                  <a href="/api/terminal/all-trades" target="_blank" rel="noopener noreferrer" className="export-trades-link">Export all</a>
                </div>
                {!trades?.length ? (
                  <p className="muted">No trades yet</p>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Price</th>
                        <th>P&L</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trades.slice(0, 20).map((t, i) => (
                        <tr key={t.id ?? i}>
                          <td>{new Date(t.timestamp).toLocaleTimeString()}</td>
                          <td>{t.symbol}</td>
                          <td className={t.side.toLowerCase()}>{t.side}</td>
                          <td>{typeof t.qty === 'number' && t.qty < 1 ? t.qty.toFixed(8).replace(/\.?0+$/, '') : t.qty}</td>
                          <td>{formatCurrency(t.price)}</td>
                          <td className={t.pnl != null ? (t.pnl >= 0 ? 'positive' : 'negative') : ''}>{t.pnl != null ? formatCurrency(t.pnl) : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </div>
        )}

        {view === 'bot' && <BotDashboard />}

        {view === 'futures' && <FuturesDashboard />}

        {view === 'backtest' && (
          <>
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
                      setSymbol(e.target.value === 'crypto' ? 'BTCUSDT' : 'AAPL')
                    }}
                    style={{ padding: '8px', fontSize: '1em', width: '100%' }}
                  >
                    <option value="equities">Stocks (Equities)</option>
                    <option value="crypto">Cryptocurrency</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>
                    <input type="checkbox" checked={useSynthetic} onChange={(e) => { setUseSynthetic(e.target.checked); if (e.target.checked) { setUseRealData(false); setUseCrypto(false) } }} />
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
                          const v = e.target.value.toUpperCase()
                          setSymbol(v)
                          if (v.length >= 1) {
                            try {
                              const res = await axios.get(`/api/symbols/search?query=${encodeURIComponent(v)}&asset_class=${assetClass}`)
                              setSymbolOptions(res.data.symbols || [])
                            } catch { setSymbolOptions([]) }
                          } else setSymbolOptions([])
                        }}
                        list="symbol-list-backtest"
                        style={{ textTransform: 'uppercase' }}
                      />
                      <datalist id="symbol-list-backtest">
                        {symbolOptions.map((o, i) => <option key={i} value={o.symbol}>{o.display}</option>)}
                      </datalist>
                    </div>
                    <div className="form-group">
                      <label>Period (Days)</label>
                      <select value={periodDays} onChange={(e) => { const d = parseInt(e.target.value); setPeriodDays(d); const end = new Date(); const start = new Date(); start.setDate(start.getDate() - d); setConfig({ ...config, start_date: start.toISOString().split('T')[0], end_date: end.toISOString().split('T')[0] }) }} style={{ padding: '8px' }}>
                        <option value={7}>Last 7 days</option>
                        <option value={30}>Last 30 days</option>
                        <option value={90}>Last 90 days</option>
                        <option value={365}>Last year</option>
                      </select>
                    </div>
                    <div className="form-group">
                      <button type="button" className="suggest-btn" onClick={fetchSuggest} disabled={suggestLoading}>
                        Get Trade Suggestion
                      </button>
                    </div>
                    {suggestions && (
                      <div className={`suggestion-card signal-${(suggestions.signal || 'HOLD').toLowerCase()}`}>
                        <h3>{suggestions.signal} {suggestions.symbol}</h3>
                        <p><strong>Confidence:</strong> {(suggestions.confidence != null ? suggestions.confidence * 100 : 0).toFixed(0)}%</p>
                        <p><strong>Current Price:</strong> {formatCurrency(suggestions.current_price)}</p>
                        <p><strong>Risk Level:</strong> {suggestions.risk_level || '—'}</p>
                        <p><strong>Expected Monthly Return:</strong> {formatPercent(suggestions.expected_monthly_return)}</p>
                        {suggestions.recommendation && <p><strong>Recommendation:</strong> {suggestions.recommendation}</p>}
                      </div>
                    )}
                  </>
                )}
                {!useSynthetic && !useRealData && (
                  <div className="form-group">
                    <label>Upload Data File (CSV or Parquet)</label>
                    <input type="file" accept=".csv,.parquet" onChange={(e) => setFile(e.target.files[0])} required={!useSynthetic && !useRealData} />
                  </div>
                )}
                <div className="form-row">
                  <div className="form-group">
                    <label>Initial Cash ($)</label>
                    <input type="number" value={config.initial_cash || ''} onChange={(e) => setConfig({ ...config, initial_cash: parseNumeric(e.target.value) || 100000 })} min="1000" step="1000" />
                  </div>
                  <div className="form-group">
                    <label>Fast SMA Period</label>
                    <input type="number" value={config.fast_period || ''} onChange={(e) => setConfig({ ...config, fast_period: parseNumeric(e.target.value) || 10 })} min="1" max="100" />
                  </div>
                  <div className="form-group">
                    <label>Slow SMA Period</label>
                    <input type="number" value={config.slow_period || ''} onChange={(e) => setConfig({ ...config, slow_period: parseNumeric(e.target.value) || 30 })} min="1" max="200" />
                  </div>
                </div>
                <button type="submit" disabled={loading} className="submit-btn">
                  {loading ? 'Running Backtest...' : 'Run Backtest'}
                </button>
              </form>
            </div>
            {results && (
              <div className="results">
                <h2>Backtest Results</h2>
                <div className="metrics-grid">
                  <div className="metric-card"><div className="metric-label">Total Return</div><div className="metric-value">{formatPercent(results.metrics?.total_return)}</div></div>
                  <div className="metric-card"><div className="metric-label">Sharpe Ratio</div><div className="metric-value">{results.metrics?.sharpe?.toFixed(2) ?? '—'}</div></div>
                  <div className="metric-card"><div className="metric-label">Max Drawdown</div><div className="metric-value negative">{formatPercent(results.metrics?.max_drawdown)}</div></div>
                  <div className="metric-card"><div className="metric-label">Win Rate</div><div className="metric-value">{formatPercent(results.metrics?.win_rate)}</div></div>
                  <div className="metric-card"><div className="metric-label">Trades</div><div className="metric-value">{results.metrics?.num_trades ?? 0}</div></div>
                </div>
                {results.equity_curve?.length > 0 && (
                  <div className="chart-section">
                    <h3>Equity Curve</h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={results.equity_curve}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="timestamp" tickFormatter={(v) => new Date(v).toLocaleDateString()} />
                        <YAxis tickFormatter={(v) => formatCurrency(v)} />
                        <Tooltip formatter={(v) => formatCurrency(v)} labelFormatter={(v) => new Date(v).toLocaleString()} />
                        <Line type="monotone" dataKey="equity" stroke="#2563eb" strokeWidth={2} name="Equity" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
                {results.trades?.length > 0 && (
                  <div className="trades-section">
                    <h3>Trades ({results.trades.length})</h3>
                    <div className="trades-table">
                      <table>
                        <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th><th>Cost</th><th>PnL</th></tr></thead>
                        <tbody>
                          {results.trades.slice(0, 50).map((t, i) => (
                            <tr key={i}>
                              <td>{new Date(t.timestamp).toLocaleString()}</td>
                              <td>{t.symbol}</td>
                              <td className={t.side?.toLowerCase()}>{t.side}</td>
                              <td>{parseFloat(t.quantity).toFixed(2)}</td>
                              <td>{formatCurrency(parseFloat(t.price))}</td>
                              <td>{formatCurrency(parseFloat(t.cost))}</td>
                              <td className={parseFloat(t.pnl) >= 0 ? 'positive' : 'negative'}>{formatCurrency(parseFloat(t.pnl))}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default App
