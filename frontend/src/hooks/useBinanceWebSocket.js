/**
 * useBinanceWebSocket - Connects to backend WS proxy, which relays Binance kline + ticker.
 * Returns { price, kline, connected } for real-time chart and price updates.
 */
import { useState, useEffect, useCallback } from 'react'

const getWsUrl = (symbol, interval) => {
  const sym = (symbol || 'btcusdt').toLowerCase().replace('/', '')
  const int = (interval || '1h').toLowerCase()
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host
  return `${protocol}//${host}/ws/crypto?symbol=${sym}&interval=${int}`
}

export function useBinanceWebSocket(symbol, interval, enabled = true) {
  const [price, setPrice] = useState(null)
  const [kline, setKline] = useState(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    if (!enabled || !symbol?.trim()) return

    const url = getWsUrl(symbol, interval)
    let ws = null
    let reconnectTimer = null
    let mounted = true

    const connect = () => {
      try {
        ws = new WebSocket(url)
        ws.onopen = () => {
          if (mounted) setConnected(true)
        }
        ws.onmessage = (event) => {
          if (!mounted) return
          try {
            const msg = JSON.parse(event.data)
            if (msg.error) return
            if (msg.event === '24hrTicker' && msg.price != null) {
              setPrice(msg.price)
            }
            if (msg.event === 'kline' && msg.kline) {
              setKline(msg.kline)
            }
          } catch (_) {}
        }
        ws.onclose = () => {
          if (mounted) setConnected(false)
          reconnectTimer = setTimeout(connect, 3000)
        }
        ws.onerror = () => {
          ws?.close()
        }
      } catch (e) {
        reconnectTimer = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      mounted = false
      if (reconnectTimer) clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [symbol, interval, enabled])

  return { price, kline, connected }
}
