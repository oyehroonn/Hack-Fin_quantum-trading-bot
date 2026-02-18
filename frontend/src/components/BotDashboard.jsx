import { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'

const CRYPTO_OPTIONS = [
  { value: 'BTCUSDT', label: 'Bitcoin (BTC)' },
  { value: 'ETHUSDT', label: 'Ethereum (ETH)' },
  { value: 'BNBUSDT', label: 'BNB' },
  { value: 'SOLUSDT', label: 'Solana (SOL)' },
  { value: 'XRPUSDT', label: 'Ripple (XRP)' },
  { value: 'ADAUSDT', label: 'Cardano (ADA)' },
  { value: 'DOGEUSDT', label: 'Dogecoin (DOGE)' },
  { value: 'AVAXUSDT', label: 'Avalanche (AVAX)' },
]

function formatCurrency(v) {
  if (v == null || isNaN(v)) return '$0.00'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(v)
}

function formatPct(v) {
  if (v == null || isNaN(v)) return '0.0%'
  return `${(v * 100).toFixed(1)}%`
}

export default function BotDashboard() {
  const [botStatus, setBotStatus] = useState(null)
  const [performance, setPerformance] = useState(null)
  const [decisions, setDecisions] = useState([])
  const [mcResult, setMcResult] = useState(null)
  const [stats, setStats] = useState(null)
  const [scanResults, setScanResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState(null)
  const [botSymbol, setBotSymbol] = useState('BTCUSDT')
  const [botAmount, setBotAmount] = useState(100)
  const [botInterval, setBotInterval] = useState(300)
  const [livePrice, setLivePrice] = useState(null)
  const [portfolio, setPortfolio] = useState(null)
  const wsRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await axios.get('/api/bot/status')
      setBotStatus(res.data)
    } catch { setBotStatus({ running: false }) }
  }, [])

  const fetchPerformance = useCallback(async () => {
    try {
      const res = await axios.get('/api/bot/performance?session_id=bot-default')
      setPerformance(res.data)
    } catch {}
  }, [])

  const fetchDecisions = useCallback(async () => {
    try {
      const res = await axios.get('/api/bot/decisions?session_id=bot-default&limit=50')
      setDecisions(res.data?.decisions ?? [])
    } catch {}
  }, [])

  const fetchPortfolio = useCallback(async () => {
    try {
      const res = await axios.get('/api/terminal/portfolio', { params: { session_id: 'bot-default' } })
      setPortfolio(res.data)
    } catch {}
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchPerformance()
    fetchDecisions()
    fetchPortfolio()
    const id = setInterval(() => { fetchStatus(); fetchPerformance(); fetchDecisions(); fetchPortfolio() }, 10000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.close()
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/crypto?symbol=${botSymbol.toLowerCase()}&interval=1m`
    const ws = new WebSocket(wsUrl)
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        if (data.price) setLivePrice(data.price)
      } catch {}
    }
    wsRef.current = ws
    return () => ws.close()
  }, [botSymbol])

  const startBot = async () => {
    setLoading(true)
    setError(null)
    try {
      await axios.post('/api/bot/start', null, {
        params: { session_id: 'bot-default', symbol: botSymbol, trade_amount: botAmount, interval: botInterval },
      })
      await fetchStatus()
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const stopBot = async () => {
    setLoading(true)
    try {
      await axios.post('/api/bot/stop')
      await fetchStatus()
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const runOnce = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post('/api/bot/run-once', null, {
        params: { session_id: 'bot-default', symbol: botSymbol, trade_amount: botAmount },
      })
      setMcResult(null)
      await fetchDecisions()
      await fetchPerformance()
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const runMC = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post('/api/bot/monte-carlo', null, { params: { symbol: botSymbol, horizon: 7, paths: 10000 } })
      setMcResult(res.data)
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const fetchStats = async () => {
    setLoading(true)
    try {
      const res = await axios.get('/api/bot/statistics', { params: { symbol: botSymbol } })
      setStats(res.data)
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const scanAllCryptos = async () => {
    setScanning(true)
    setError(null)
    setScanResults(null)
    try {
      const res = await axios.get('/api/bot/scan-all')
      setScanResults(res.data?.results ?? [])
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setScanning(false) }
  }

  const isRunning = botStatus?.running
  const cash = portfolio?.cash ?? 0
  const equity = portfolio?.equity ?? 0

  return (
    <div className="bot-dashboard">
      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
          <button type="button" className="dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="bot-controls">
        <div className="bot-header">
          <h2>Autonomous Trading Bot</h2>
          <div className="bot-portfolio-summary">
            <span className="portfolio-item">Cash: <strong>{formatCurrency(cash)}</strong></span>
            <span className="portfolio-item">Equity: <strong>{formatCurrency(equity)}</strong></span>
          </div>
        </div>
        <div className="bot-status-badge" style={{ color: isRunning ? '#22c55e' : '#94a3b8' }}>
          {isRunning ? '● Running' : '○ Stopped'}
          {isRunning && botStatus?.symbol && <span> — {botStatus.symbol} (${botStatus.trade_amount}/trade, {botStatus.trades_executed} trades)</span>}
        </div>

        <div className="bot-config">
          <div className="bot-config-row">
            <label>Cryptocurrency</label>
            <select value={botSymbol} onChange={(e) => setBotSymbol(e.target.value)} disabled={isRunning} className="crypto-select">
              {CRYPTO_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div className="bot-config-row">
            <label>Live Price</label>
            <div className="live-price-display">
              {livePrice ? formatCurrency(livePrice) : '—'}
              <span className="live-dot">●</span>
            </div>
          </div>
          <div className="bot-config-row">
            <label>Trade Amount ($)</label>
            <input type="number" value={botAmount} onChange={(e) => setBotAmount(Number(e.target.value))} min={10} disabled={isRunning} />
          </div>
          <div className="bot-config-row">
            <label>Check Interval (sec)</label>
            <input type="number" value={botInterval} onChange={(e) => setBotInterval(Number(e.target.value))} min={30} disabled={isRunning} />
          </div>
        </div>

        <div className="bot-actions">
          {!isRunning ? (
            <button className="btn-bot-start" onClick={startBot} disabled={loading}>Start Bot</button>
          ) : (
            <button className="btn-bot-stop" onClick={stopBot} disabled={loading}>Stop Bot</button>
          )}
          <button className="btn-bot-once" onClick={runOnce} disabled={loading || isRunning}>Run Once</button>
          <button className="btn-bot-scan" onClick={scanAllCryptos} disabled={scanning}>
            {scanning ? 'Scanning...' : 'Scan All Cryptos'}
          </button>
          <button className="btn-bot-mc" onClick={runMC} disabled={loading}>Monte Carlo</button>
          <button className="btn-bot-stats" onClick={fetchStats} disabled={loading}>Statistics</button>
        </div>
      </div>

      {scanResults && scanResults.length > 0 && (
        <div className="bot-scan-results">
          <h3>Crypto Scan Results — Best Opportunities</h3>
          <div className="scan-table">
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Price</th>
                  <th>Signal</th>
                  <th>Confidence</th>
                  <th>Expected Return</th>
                  <th>VaR 95%</th>
                  <th>P(Profit)</th>
                  <th>Regime</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {scanResults.map((r) => (
                  <tr key={r.symbol} className={r.signal === 'BUY' ? 'scan-buy' : r.signal === 'SELL' ? 'scan-sell' : ''}>
                    <td><strong>{r.symbol}</strong></td>
                    <td>{formatCurrency(r.price)}</td>
                    <td className={r.signal === 'BUY' ? 'buy' : r.signal === 'SELL' ? 'sell' : ''}>{r.signal}</td>
                    <td>{formatPct(r.confidence)}</td>
                    <td className={r.expected_return >= 0 ? 'positive' : 'negative'}>{formatPct(r.expected_return)}</td>
                    <td className="negative">{formatPct(r.var_95)}</td>
                    <td>{formatPct(r.prob_profit)}</td>
                    <td>{r.regime}</td>
                    <td>
                      <button className="btn-select-crypto" onClick={() => setBotSymbol(r.symbol)}>Select</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {performance && performance.total_trades > 0 && (
        <div className="bot-perf">
          <h3>Performance</h3>
          <div className="metrics-grid">
            <div className="metric-card"><div className="metric-label">Total Trades</div><div className="metric-value">{performance.total_trades}</div></div>
            <div className="metric-card"><div className="metric-label">Win Rate</div><div className="metric-value">{formatPct(performance.win_rate)}</div></div>
            <div className="metric-card"><div className="metric-label">Total P&L</div><div className={'metric-value ' + ((performance.total_pnl ?? 0) >= 0 ? 'positive' : 'negative')}>{formatCurrency(performance.total_pnl)}</div></div>
            <div className="metric-card"><div className="metric-label">Avg Trade P&L</div><div className="metric-value">{formatCurrency(performance.avg_trade_pnl)}</div></div>
            <div className="metric-card"><div className="metric-label">Best Trade</div><div className="metric-value positive">{formatCurrency(performance.best_trade_pnl)}</div></div>
            <div className="metric-card"><div className="metric-label">Worst Trade</div><div className="metric-value negative">{formatCurrency(performance.worst_trade_pnl)}</div></div>
            <div className="metric-card"><div className="metric-label">Max Drawdown</div><div className="metric-value negative">{formatPct(performance.max_drawdown)}</div></div>
            <div className="metric-card"><div className="metric-label">Portfolio Value</div><div className="metric-value">{formatCurrency(performance.portfolio_value)}</div></div>
          </div>
        </div>
      )}

      {mcResult && (
        <div className="bot-mc-result">
          <h3>Monte Carlo Simulation — {mcResult.symbol}</h3>
          <div className="metrics-grid">
            <div className="metric-card"><div className="metric-label">Current Price</div><div className="metric-value">{formatCurrency(mcResult.current_price)}</div></div>
            <div className="metric-card"><div className="metric-label">Expected Price ({mcResult.horizon_days}d)</div><div className="metric-value">{formatCurrency(mcResult.expected_price)}</div></div>
            <div className="metric-card"><div className="metric-label">Expected Return</div><div className={'metric-value ' + (mcResult.expected_return >= 0 ? 'positive' : 'negative')}>{formatPct(mcResult.expected_return)}</div></div>
            <div className="metric-card"><div className="metric-label">VaR 95%</div><div className="metric-value negative">{formatPct(mcResult.var_95)}</div></div>
            <div className="metric-card"><div className="metric-label">CVaR 95%</div><div className="metric-value negative">{formatPct(mcResult.cvar_95)}</div></div>
            <div className="metric-card"><div className="metric-label">P(Profit)</div><div className="metric-value">{formatPct(mcResult.prob_profit)}</div></div>
            <div className="metric-card"><div className="metric-label">5th Percentile</div><div className="metric-value">{formatCurrency(mcResult.percentile_5)}</div></div>
            <div className="metric-card"><div className="metric-label">95th Percentile</div><div className="metric-value">{formatCurrency(mcResult.percentile_95)}</div></div>
          </div>
        </div>
      )}

      {stats && (
        <div className="bot-stats-result">
          <h3>Statistical Analysis</h3>
          <div className="metrics-grid">
            <div className="metric-card"><div className="metric-label">Price</div><div className="metric-value">{formatCurrency(stats.current_price)}</div></div>
            <div className="metric-card"><div className="metric-label">Annualized Return</div><div className={'metric-value ' + (stats.annualized_return >= 0 ? 'positive' : 'negative')}>{formatPct(stats.annualized_return)}</div></div>
            <div className="metric-card"><div className="metric-label">Ann. Volatility</div><div className="metric-value">{formatPct(stats.annualized_volatility)}</div></div>
            <div className="metric-card"><div className="metric-label">Sharpe Ratio</div><div className="metric-value">{stats.sharpe_ratio?.toFixed(2)}</div></div>
            <div className="metric-card"><div className="metric-label">RSI</div><div className="metric-value">{stats.rsi?.toFixed(1)}</div></div>
            <div className="metric-card"><div className="metric-label">Regime</div><div className="metric-value">{stats.regime}</div></div>
            <div className="metric-card"><div className="metric-label">Trend</div><div className="metric-value">{(stats.trend_strength * 100)?.toFixed(2)}%</div></div>
            <div className="metric-card"><div className="metric-label">Max Drawdown</div><div className="metric-value negative">{formatPct(stats.max_drawdown)}</div></div>
          </div>
        </div>
      )}

      <div className="bot-decisions">
        <h3>Decision Log ({decisions.length})</h3>
        {decisions.length === 0 ? (
          <p className="muted">No decisions yet. Start the bot or click "Run Once".</p>
        ) : (
          <div className="trades-table">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Action</th>
                  <th>Price</th>
                  <th>Amount</th>
                  <th>ML Signal</th>
                  <th>ML Conf</th>
                  <th>MC VaR</th>
                  <th>Regime</th>
                  <th>Reasoning</th>
                </tr>
              </thead>
              <tbody>
                {decisions.slice(0, 50).map((d, i) => (
                  <tr key={d.id ?? i}>
                    <td>{new Date(d.timestamp).toLocaleString()}</td>
                    <td className={d.action === 'BUY' ? 'buy' : d.action === 'SELL' ? 'sell' : ''}>{d.action}</td>
                    <td>{formatCurrency(d.price)}</td>
                    <td>{d.action === 'BUY' ? formatCurrency(d.amount_usd) : d.qty ? d.qty.toFixed(6) : '—'}</td>
                    <td>{d.ml_signal}</td>
                    <td>{d.ml_confidence != null ? `${(d.ml_confidence * 100).toFixed(0)}%` : '—'}</td>
                    <td className="negative">{d.monte_carlo_var != null ? formatPct(d.monte_carlo_var) : '—'}</td>
                    <td>{d.regime || '—'}</td>
                    <td className="reasoning-cell" title={d.reasoning}>{d.reasoning?.slice(0, 60)}{d.reasoning?.length > 60 ? '…' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
