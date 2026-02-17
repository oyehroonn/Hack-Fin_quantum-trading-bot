import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

/**
 * Polls /api/market/price for latest price. Returns { price, loading, error, refresh }.
 */
export function useLivePrice(symbol, assetClass, intervalMs = 5000) {
  const [price, setPrice] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchPrice = useCallback(async () => {
    if (!symbol?.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await axios.get('/api/market/price', {
        params: { symbol: symbol.toUpperCase().replace('/', ''), asset_class: assetClass },
      })
      setPrice(res.data?.price ?? null)
    } catch (err) {
      setError(err.response?.data?.detail ?? err.message)
    } finally {
      setLoading(false)
    }
  }, [symbol, assetClass])

  useEffect(() => {
    fetchPrice()
  }, [fetchPrice])

  useEffect(() => {
    if (!symbol?.trim()) return
    const id = setInterval(fetchPrice, intervalMs)
    return () => clearInterval(id)
  }, [symbol, assetClass, intervalMs, fetchPrice])

  return { price, loading, error, refresh: fetchPrice }
}
