import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const STORAGE_KEY = 'quantum_terminal_session_id'
const DEFAULT_CASH = 100000

/**
 * Manages terminal session: create/load from localStorage, portfolio, positions, trades.
 * Returns { sessionId, portfolio, positions, trades, loading, error, executeTrade, refetch, createNewSession }.
 */
export function useTerminalSession() {
  const [sessionId, setSessionId] = useState(null)
  const [portfolio, setPortfolio] = useState(null)
  const [positions, setPositions] = useState([])
  const [trades, setTrades] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const createNewSession = useCallback(async (initialCash = DEFAULT_CASH) => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post('/api/terminal/session', null, {
        params: { initial_cash: initialCash },
      })
      const sid = res.data?.session_id
      if (sid) {
        setSessionId(sid)
        try { localStorage.setItem(STORAGE_KEY, sid) } catch {}
      }
      return sid
    } catch (err) {
      setError(err.response?.data?.detail ?? err.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const refetch = useCallback(async (sid, priceMap = null) => {
    const id = sid ?? sessionId
    if (!id) return
    const pricesParam = priceMap && Object.keys(priceMap).length
      ? '&prices=' + encodeURIComponent(Object.entries(priceMap).map(([k, v]) => `${k}:${v}`).join(','))
      : ''
    try {
      const [portRes, posRes, tradesRes] = await Promise.all([
        axios.get(`/api/terminal/portfolio?session_id=${id}${pricesParam}`),
        axios.get(`/api/terminal/positions?session_id=${id}`),
        axios.get(`/api/terminal/trades?session_id=${id}&limit=100`),
      ])
      setPortfolio(portRes.data)
      setPositions(posRes.data?.positions ?? [])
      setTrades(tradesRes.data?.trades ?? [])
    } catch (err) {
      if (err.response?.status === 404) {
        try { localStorage.removeItem(STORAGE_KEY) } catch {}
        setSessionId(null)
      }
      setError(err.response?.data?.detail ?? err.message)
    }
  }, [sessionId])

  const loadOrCreateSession = useCallback(async () => {
    let sid = null
    try { sid = localStorage.getItem(STORAGE_KEY) } catch {}
    if (sid) {
      try {
        await axios.get(`/api/terminal/session/${sid}`)
        setSessionId(sid)
        return sid
      } catch {
        sid = null
      }
    }
    try {
      const res = await axios.get('/api/terminal/default-session')
      const defaultId = res.data?.session_id
      if (defaultId) {
        setSessionId(defaultId)
        try { localStorage.setItem(STORAGE_KEY, defaultId) } catch {}
        await refetch(defaultId)
        return defaultId
      }
    } catch {}
    const newSid = await createNewSession()
    if (newSid) await refetch(newSid)
    return newSid
  }, [createNewSession, refetch])

  const restoreSession = useCallback(async (sid) => {
    if (!sid?.trim()) return false
    setLoading(true)
    setError(null)
    try {
      await axios.get(`/api/terminal/session/${sid}`)
      setSessionId(sid)
      try { localStorage.setItem(STORAGE_KEY, sid) } catch {}
      await refetch(sid)
      return true
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Session not found')
      return false
    } finally {
      setLoading(false)
    }
  }, [refetch])

  useEffect(() => {
    loadOrCreateSession()
  }, [])

  useEffect(() => {
    if (sessionId) refetch()
  }, [sessionId])

  const executeTrade = useCallback(async ({ symbol, side, qty, price, asset_class }) => {
    if (!sessionId || !symbol || !side || !qty || !price) return null
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post('/api/terminal/trade', {
        session_id: sessionId,
        symbol: symbol.toUpperCase().replace('/', ''),
        side: side.toUpperCase(),
        qty: Number(qty),
        price: Number(price),
        asset_class: asset_class || 'equities',
      })
      await refetch()
      return res.data
    } catch (err) {
      setError(err.response?.data?.detail ?? err.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [sessionId, refetch])

  const clearError = useCallback(() => setError(null), [])

  return {
    sessionId,
    portfolio,
    positions,
    trades,
    loading,
    error,
    clearError,
    executeTrade,
    refetch: () => refetch(sessionId),
    createNewSession,
    restoreSession,
  }
}
