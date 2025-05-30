import paho.mqtt.client as mqtt
import json
import logging
import time

# Setup logger
logging.basicConfig(level=logging.DEBUG, format="[DEBUG] %(message)s")
logger = logging.getLogger(__name__)

class MQTTThingsBoardClient:
    def __init__(self):
        """Initialize MQTT client."""
        self.client = None
        self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the client receives a CONNACK response."""
        if rc == 0:
            self.connected = True
            logger.debug("Connected to ThingsBoard server")
        else:
            self.connected = False
            error_msg = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorized"
            }.get(rc, f"Connection failed with code {rc}")
            logger.error(error_msg)

    def send_telemetry(self, server_IP, port, token, key, value, retries=3, retry_delay=2):
        """
        Send telemetry data to ThingsBoard server via MQTT.

        Args:
            server_IP (str): Domain name of the ThingsBoard server (e.g., 'app.coreiot.io').
            port (int): MQTT port for connection (e.g., 1883).
            token (str): Device access token for authentication.
            key (str): Telemetry key (e.g., 'temperature').
            value: Telemetry value (e.g., 25.5).
            retries (int): Number of connection retries (default: 3).
            retry_delay (int): Delay between retries in seconds (default: 2).

        Returns:
            bool: True if data sent successfully, False otherwise.
        """
        for attempt in range(1, retries + 1):
            try:
                # Initialize client if not connected
                if self.client is None or not self.connected:
                    logger.debug(f"Attempt {attempt}/{retries}: Connecting to {server_IP}:{port} with token {token}")
                    self.client = mqtt.Client()
                    self.client.username_pw_set(token)
                    self.client.on_connect = self.on_connect
                    self.client.connect(server_IP, port, 60)
                    self.client.loop_start()
                    # Wait for connection
                    timeout = time.time() + 5
                    while not self.connected and time.time() < timeout:
                        time.sleep(0.1)
                    if not self.connected:
                        raise Exception("Connection timeout")

                # Send telemetry
                payload = {key: value}
                result = self.client.publish("v1/devices/me/telemetry", json.dumps(payload), qos=1)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.debug(f"Sent telemetry: {key}={value}")
                    return True
                else:
                    logger.error(f"Failed to publish telemetry, return code: {result.rc}")
                    return False

            except Exception as e:
                logger.error(f"Attempt {attempt}/{retries} failed: {str(e)}")
                if "not authorized" in str(e).lower() or "Connection refused" in str(e):
                    logger.error("Invalid token or device not authorized. Please check the access token in ThingsBoard.")
                if self.client is not None:
                    self.client.loop_stop()
                    self.client.disconnect()
                    self.client = None
                    self.connected = False
                if attempt < retries:
                    logger.debug(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Failed to send telemetry.")
                    return False

    def disconnect(self):
        """Disconnect from ThingsBoard server."""
        if self.client is not None:
            if self.connected:
                self.client.loop_stop()
                self.client.disconnect()
                logger.debug("Disconnected from ThingsBoard server")
            self.client = None
            self.connected = False