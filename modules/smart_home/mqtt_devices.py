"""MQTT device control — Stella commands networked devices (relays, lights,
plugs) on RobotNet through the local mosquitto broker.

A device is a friendly name mapped to an MQTT base topic, e.g.
    light -> robotnet/light
Stella publishes "ON"/"OFF" (retained) to  <base>/set  and listens on
<base>/state for the device's reported state. An ESP32/ESP8266 (Tasmota,
ESPHome, or the sample sketch in docs) subscribes to <base>/set and drives a
relay. Fully offline-capable — the broker runs on the Pi.
"""

from __future__ import annotations

import logging

try:
    import paho.mqtt.client as mqtt
    PAHO_OK = True
except Exception:  # pragma: no cover
    mqtt = None
    PAHO_OK = False

logger = logging.getLogger("MQTT")


class MqttDevices:
    def __init__(self, broker: str = "localhost", port: int = 1883,
                 devices: dict | None = None):
        self.broker = broker
        self.port = int(port)
        self.devices = {str(k): str(v) for k, v in (devices or {}).items()}
        self.state: dict[str, str] = {}
        self.connected = False
        self.enabled = PAHO_OK
        self._client = None
        if PAHO_OK:
            self._start()
        else:
            logger.info("paho-mqtt not installed — device control disabled")

    # -- lifecycle ---------------------------------------------------------
    def _start(self):
        try:
            self._client = mqtt.Client()
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)
            self._client.connect_async(self.broker, self.port, keepalive=30)
            self._client.loop_start()   # background thread; auto-reconnects
            logger.info("MQTT device hub starting (%s:%d, %d device(s))",
                        self.broker, self.port, len(self.devices))
        except Exception as exc:
            logger.warning("MQTT start failed: %s", exc)

    def _on_connect(self, client, userdata, flags, rc, *a):
        self.connected = (rc == 0)
        if self.connected:
            logger.info("MQTT connected — subscribing to %d device state topics",
                        len(self.devices))
            for base in self.devices.values():
                client.subscribe(base + "/state")
        else:
            logger.warning("MQTT connect failed rc=%s", rc)

    def _on_disconnect(self, *a):
        self.connected = False

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode(errors="ignore").strip().upper()
        for name, base in self.devices.items():
            if msg.topic == base + "/state":
                self.state[name] = payload

    # -- public API --------------------------------------------------------
    def _match(self, name: str):
        n = (name or "").lower().strip()
        if not n:
            return None
        for k in self.devices:
            kl = k.lower()
            if kl == n or kl in n or n in kl:
                return k
        return None

    def known(self, name: str) -> bool:
        return self._match(name) is not None

    def list_devices(self):
        return list(self.devices.keys())

    def set(self, name: str, on: bool) -> bool:
        """Publish ON/OFF (retained) to the device's /set topic. True if sent."""
        if not (self.enabled and self._client):
            return False
        key = self._match(name)
        if not key:
            return False
        topic = self.devices[key] + "/set"
        payload = "ON" if on else "OFF"
        try:
            self._client.publish(topic, payload, qos=1, retain=True)
            self.state[key] = payload
            logger.info("device '%s' -> %s (%s)", key, payload, topic)
            return True
        except Exception as exc:
            logger.warning("publish failed: %s", exc)
            return False

    def get_state(self, name: str) -> str:
        key = self._match(name)
        return self.state.get(key, "unknown") if key else "unknown"

    def stop(self):
        try:
            if self._client:
                self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass
