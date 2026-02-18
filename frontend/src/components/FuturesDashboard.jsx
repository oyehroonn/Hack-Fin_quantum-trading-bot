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

export default function FuturesDashboard() {
  const [botStatus, setBotStatus] = useState(null)
  const [session, setSession] = useState(null)
  const [positions, setPositions] = useState([])
  const [trades, setTrades] = useState([])
  const [scanResults, setScanResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [scanning, setScanning] = useState(false)
  const [error, setError] = useState(null)
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [baseMargin, setBaseMargin] = useState(100)
  const [maxLeverage, setMaxLeverage] = useState(50)
  const [interval, setInterval] = useState(60)
  const [livePrice, setLivePrice] = useState(null)
  const wsRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await axios.get('/api/futures/status')
      setBotStatus(res.data)
    } catch { setBotStatus({ running: false }) }
  }, [])

  const fetchSession = useCallback(async () => {
    try {
      const res = await axios.get('/api/futures/session?session_id=futures-default')
      setSession(res.data)
    } catch {}
  }, [])

  const fetchPositions = useCallback(async () => {
    try {
      const res = await axios.get('/api/futures/positions?session_id=futures-default&status=OPEN')
      setPositions(res.data?.positions ?? [])
    } catch {}
  }, [])

  const fetchTrades = useCallback(async () => {
    try {
      const res = await axios.get('/api/futures/trades?session_id=futures-default&limit=30')
      setTrades(res.data?.trades ?? [])
    } catch {}
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchSession()
    fetchPositions()
    fetchTrades()
    const id = window.setInterval(() => {
      fetchStatus()
      fetchSession()
      fetchPositions()
      fetchTrades()
    }, 8000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    if (wsRef.current) wsRef.current.close()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/crypto?symbol=${symbol.toLowerCase()}&interval=1m`
    const ws = new WebSocket(wsUrl)
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        if (data.price) setLivePrice(data.price)
      } catch {}
    }
    wsRef.current = ws
    return () => ws.close()
  }, [symbol])

  const startBot = async () => {
    setLoading(true)
    setError(null)
    try {
      await axios.post('/api/futures/start', null, {
        params: { session_id: 'futures-default', symbol, base_margin: baseMargin, max_leverage: maxLeverage, interval },
      })
      await fetchStatus()
      await fetchSession()
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const stopBot = async () => {
    setLoading(true)
    try {
      await axios.post('/api/futures/stop')
      await fetchStatus()
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const runOnce = async () => {
    setLoading(true)
    setError(null)
    try {
      await axios.post('/api/futures/run-once', null, {
        params: { session_id: 'futures-default', symbol, base_margin: baseMargin, max_leverage: maxLeverage },
      })
      await fetchPositions()
      await fetchTrades()
      await fetchSession()
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const scanAll = async () => {
    setScanning(true)
    setError(null)
    try {
      const res = await axios.get('/api/futures/scan', { params: { max_leverage: maxLeverage } })
      setScanResults(res.data?.results ?? [])
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setScanning(false) }
  }

  const closePosition = async (positionId) => {
    setLoading(true)
    try {
      await axios.post(`/api/futures/close-position/${positionId}`)
      await fetchPositions()
      await fetchTrades()
      await fetchSession()
    } catch (e) { setError(e.response?.data?.detail ?? e.message) }
    finally { setLoading(false) }
  }

  const isRunning = botStatus?.running
  const margin = session?.current_margin ?? 0
  const totalPnl = session?.total_pnl ?? 0
  const winRate = session?.total_trades > 0 ? (session.winning_trades / session.total_trades) : 0

  return (
    <div className="futures-dashboard">
      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
          <button type="button" className="dismiss" onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="futures-controls">
        <div className="futures-header">
          <h2>Futures Trading Bot (Simulated)</h2>
          <div className="futures-summary">
            <span className="summary-item">Margin: <strong>{formatCurrency(margin)}</strong></span>
            <span className="summary-item">P&L: <strong className={totalPnl >= 0 ? 'positive' : 'negative'}>{formatCurrency(totalPnl)}</strong></span>
            <span className="summary-item">Win Rate: <strong>{formatPct(winRate)}</strong></span>
            <span className="summary-item">Liquidations: <strong className="negative">{session?.liquidations ?? 0}</strong></span>
          </div>
        </div>

        <div className="futures-status-badge" style={{ color: isRunning ? '#22c55e' : '#94a3b8' }}>
          {isRunning ? '● Running' : '○ Stopped'}
          {isRunning && botStatus?.symbol && (
            <span> — {botStatus.symbol} ({botStatus.max_leverage}x max, {botStatus.trades_executed} trades)</span>
          )}
        </div>

        <div className="futures-config">
          <div className="config-row">
            <label>Cryptocurrency</label>
            <select value={symbol} onChange={(e) => setSymbol(e.target.value)} disabled={isRunning} className="crypto-select">
              {CRYPTO_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div className="config-row">
            <label>Live Price</label>
            <div className="live-price-display">
              {livePrice ? formatCurrency(livePrice) : '—'}
              <span className="live-dot">●</span>
            </div>
          </div>
          <div className="config-row">
            <label>Base Margin ($)</label>
            <input type="number" value={baseMargin} onChange={(e) => setBaseMargin(Number(e.target.value))} min={10} disabled={isRunning} />
          </div>
          <div className="config-row">
            <label>Max Leverage</label>
            <select value={maxLeverage} onChange={(e) => setMaxLeverage(Number(e.target.value))} disabled={isRunning}>
              <option value={10}>10x</option>
              <option value={25}>25x</option>
              <option value={50}>50x</option>
              <option value={75}>75x</option>
              <option value={100}>100x</option>
            </select>
          </div>
          <div className="config-row">
            <label>Interval (sec)</label>
            <input type="number" value={interval} onChange={(e) => setInterval(Number(e.target.value))} min={30} disabled={isRunning} />
          </div>
        </div>

        <div className="futures-actions">
          {!isRunning ? (
            <button className="btn-futures-start" onClick={startBot} disabled={loading}>Start Futures Bot</button>
          ) : (
            <button className="btn-futures-stop" onClick={stopBot} disabled={loading}>Stop Bot</button>
          )}
          <button className="btn-futures-once" onClick={runOnce} disabled={loading || isRunning}>Run Once</button>
          <button className="btn-futures-scan" onClick={scanAll} disabled={scanning}>
            {scanning ? 'Scanning...' : 'Scan Opportunities'}
          </button>
        </div>
      </div>

      {positions.length > 0 && (
        <div className="futures-positions">
          <h3>Open Positions ({positions.length})</h3>
          <div className="positions-grid">
            {positions.map((p) => {
              const pnlPct = p.margin > 0 ? (p.unrealized_pnl / p.margin) : 0
              return (
                <div key={p.id} className={`position-card ${p.direction.toLowerCase()}`}>
                  <div className="position-header">
                    <span className={`direction-badge ${p.direction.toLowerCase()}`}>{p.direction}</span>
                    <span className="leverage-badge">{p.leverage}x</span>
                    <span className="symbol">{p.symbol}</span>
                  </div>
                  <div className="position-details">
                    <div className="detail-row">
                      <span>Entry</span>
                      <span>{formatCurrency(p.entry_price)}</span>
                    </div>
                    <div className="detail-row">
                      <span>Current</span>
                      <span>{formatCurrency(p.current_price)}</span>
                    </div>
                    <div className="detail-row">
                      <span>Margin</span>
                      <span>{formatCurrency(p.margin)}</span>
                    </div>
                    <div className="detail-row">
                      <span>Size</span>
                      <span>{formatCurrency(p.position_size)}</span>
                    </div>
                    <div className="detail-row">
                      <span>Liquidation</span>
                      <span className="negative">{formatCurrency(p.liquidation_price)}</span>
                    </div>
                    <div className="detail-row pnl-row">
                      <span>Unrealized P&L</span>
                      <span className={p.unrealized_pnl >= 0 ? 'positive' : 'negative'}>
                        {formatCurrency(p.unrealized_pnl)} ({formatPct(pnlPct)})
                      </span>
                    </div>
                  </div>
                  <button className="btn-close-position" onClick={() => closePosition(p.id)} disabled={loading}>
                    Close Position
                  </button>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {scanResults && scanResults.length > 0 && (
        <div className="futures-scan-results">
          <h3>Futures Opportunities — LONG & SHORT Signals</h3>
          <div className="scan-table">
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Price</th>
                  <th>Signal</th>
                  <th>Confidence</th>
                  <th>Leverage</th>
                  <th>Risk Score</th>
                  <th>Expected Return</th>
                  <th>VaR 95%</th>
                  <th>Regime</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {scanResults.map((r) => (
                  <tr key={r.symbol} className={r.signal === 'LONG' ? 'scan-long' : r.signal === 'SHORT' ? 'scan-short' : ''}>
                    <td><strong>{r.symbol}</strong></td>
                    <td>{formatCurrency(r.price)}</td>
                    <td className={r.signal === 'LONG' ? 'long' : r.signal === 'SHORT' ? 'short' : ''}>{r.signal}</td>
                    <td>{formatPct(r.confidence)}</td>
                    <td><strong>{r.leverage}x</strong></td>
                    <td>{r.risk_score.toFixed(2)}</td>
                    <td className={r.expected_return >= 0 ? 'positive' : 'negative'}>{formatPct(r.expected_return)}</td>
                    <td className="negative">{formatPct(r.var_95)}</td>
                    <td>{r.regime}</td>
                    <td>
                      <button className="btn-select-crypto" onClick={() => setSymbol(r.symbol)}>Select</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="futures-trades">
        <h3>Recent Trades ({trades.length})</h3>
        {trades.length === 0 ? (
          <p className="muted">No trades yet. Start the bot or click "Run Once".</p>
        ) : (
          <div className="trades-table">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Symbol</th>
                  <th>Action</th>
                  <th>Price</th>
                  <th>Leverage</th>
                  <th>Margin</th>
                  <th>Size</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice(0, 30).map((t, i) => (
                  <tr key={t.id ?? i}>
                    <td>{new Date(t.timestamp).toLocaleString()}</td>
                    <td>{t.symbol}</td>
                    <td className={t.action.includes('LONG') ? 'long' : 'short'}>{t.action}</td>
                    <td>{formatCurrency(t.price)}</td>
                    <td>{t.leverage}x</td>
                    <td>{formatCurrency(t.margin)}</td>
                    <td>{formatCurrency(t.position_size)}</td>
                    <td className={t.pnl >= 0 ? 'positive' : 'negative'}>{t.pnl !== 0 ? formatCurrency(t.pnl) : '—'}</td>
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
