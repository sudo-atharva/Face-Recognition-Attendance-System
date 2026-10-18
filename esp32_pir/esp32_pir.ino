// ESP32 + 4x PIR motion sensors -> lab PC over USB serial (UART).
// The Lab Monitor keeps its camera and face model asleep until this sends MOTION.
//
// Wiring (HC-SR501):  VCC -> 5V (VIN),  GND -> GND
// Wiring (AM312):     VCC -> 3V3,       GND -> GND
//   PIR 1 OUT -> GPIO 13     PIR 3 OUT -> GPIO 27
//   PIR 2 OUT -> GPIO 14     PIR 4 OUT -> GPIO 26
// HC-SR501 output is 3.3 V, safe for the ESP32. Set its "time" pot to minimum
// and jumper to repeat-trigger (H); the PC handles how long to stay awake.
// Unplugged sensors read LOW (internal pull-down), so fewer than 4 is fine.
//
// Upload with Arduino IDE (board: "ESP32 Dev Module"), then in the Lab Monitor
// website: Settings -> PIR sensor -> Port = the ESP32's COM port, baud 115200.
//
// PROTOCOL for the PC (one line each, 115200 baud) - UNCHANGED:
//   READY   after the 30 s sensor warm-up
//   MOTION  when any sensor triggers, then every 2 s while motion continues
//   ALIVE   heartbeat every 10 s (if the PC hears nothing for 30 s it keeps the camera on)
//
// SERIAL MONITOR (human-readable) - all these lines start with "# " and never
// contain the uppercase words READY / MOTION / ALIVE, so the PC ignores them.
//   - startup banner + wiring summary
//   - warm-up countdown
//   - timestamped events: "# [00:01:23] PIR 2 (GPIO 14) -> motion"
//   - live status table every 10 s (can be turned off)
//
// SERIAL MONITOR COMMANDS (type in the box, press Enter, "Newline" or "Both NL & CR"):
//   s  print status table now
//   r  reset trigger counters
//   v  toggle periodic status table on/off
//   q  quiet mode: hide all "# " lines (protocol lines still sent)
//   h  help
//   NOTE: close the Serial Monitor before the Lab Monitor website opens the
//   COM port - only one program can use the port at a time.

const uint8_t NUM_PIR = 4;
const uint8_t pirPins[NUM_PIR] = {13, 14, 27, 26};
const int LED_PIN = 2;                    // onboard LED on most ESP32 dev boards
const unsigned long WARMUP_MS = 30000;    // PIRs give false triggers while settling
const unsigned long REPEAT_MS = 2000;
const unsigned long HEARTBEAT_MS = 10000;
const unsigned long STATUS_MS = 10000;

bool lastState[NUM_PIR];
unsigned long motionStart[NUM_PIR];
unsigned long lastTrigger[NUM_PIR];       // millis() of last trigger, 0 = never
uint32_t triggerCount[NUM_PIR];
unsigned long lastMotionMsg = 0;
unsigned long lastBeat = 0;
unsigned long lastStatus = 0;
bool wasAny = false;
bool showStatus = true;
bool quiet = false;

// ---------- Serial Monitor helpers ----------

// Print a "# " info line only when not in quiet mode
void info(const char *fmt, ...) {
  if (quiet) return;
  char buf[160];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  Serial.print("# ");
  Serial.println(buf);
}

// Format uptime as HH:MM:SS
const char *stamp(unsigned long ms) {
  static char buf[12];
  unsigned long s = ms / 1000;
  snprintf(buf, sizeof(buf), "%02lu:%02lu:%02lu", s / 3600, (s / 60) % 60, s % 60);
  return buf;
}

void printHelp() {
  info("commands: s=status  r=reset counters  v=toggle status table  q=quiet  h=help");
}

void printStatus(unsigned long now) {
  if (quiet) return;
  info("---------------- status @ %s ----------------", stamp(now));
  info("PIR  GPIO  state   triggers  last trigger");
  for (uint8_t i = 0; i < NUM_PIR; i++) {
    char last[24];
    if (lastTrigger[i] == 0) snprintf(last, sizeof(last), "never");
    else snprintf(last, sizeof(last), "%lu s ago", (now - lastTrigger[i]) / 1000);
    info(" %d    %2d   %-6s  %8lu  %s", i + 1, pirPins[i],
         lastState[i] ? "ACTIVE" : "idle", (unsigned long)triggerCount[i], last);
  }
  info("------------------------------------------------");
}

void handleCommands(unsigned long now) {
  while (Serial.available()) {
    char c = Serial.read();
    switch (c) {
      case 's': case 'S':
        printStatus(now);
        break;
      case 'r': case 'R':
        for (uint8_t i = 0; i < NUM_PIR; i++) { triggerCount[i] = 0; lastTrigger[i] = 0; }
        info("counters reset");
        break;
      case 'v': case 'V':
        showStatus = !showStatus;
        info("periodic status table %s", showStatus ? "on" : "off");
        break;
      case 'q': case 'Q':
        if (quiet) { quiet = false; info("quiet mode off"); }
        else { info("quiet mode on (send q again to undo)"); quiet = true; }
        break;
      case 'h': case 'H': case '?':
        printHelp();
        break;
      default:
        break;   // ignore newlines and anything else
    }
  }
}

// ---------- setup / loop ----------

void setup() {
  Serial.begin(115200);
  delay(300);                                     // let USB-serial settle
  pinMode(LED_PIN, OUTPUT);
  for (uint8_t i = 0; i < NUM_PIR; i++) pinMode(pirPins[i], INPUT_PULLDOWN);

  info("================================================");
  info(" ESP32 4x PIR sensor node  |  115200 baud");
  info(" PIR1=GPIO13  PIR2=GPIO14  PIR3=GPIO27  PIR4=GPIO26");
  info("================================================");
  printHelp();
  info("warming up sensors for %lu s...", WARMUP_MS / 1000);

  unsigned long t0 = millis();
  unsigned long lastCountdown = 0;
  while (millis() - t0 < WARMUP_MS) {
    unsigned long el = millis() - t0;
    digitalWrite(LED_PIN, (millis() / 250) % 2);   // blink during warm-up
    if (el - lastCountdown >= 5000) {
      lastCountdown = el;
      info("warm-up: %lu s left", (WARMUP_MS - el + 999) / 1000);
    }
    handleCommands(millis());
    delay(10);
  }
  digitalWrite(LED_PIN, LOW);

  info("warm-up done, sensors armed");
  Serial.println("READY");
  unsigned long now = millis();
  lastBeat = now;
  lastStatus = now;
}

void loop() {
  unsigned long now = millis();
  bool any = false;

  handleCommands(now);

  for (uint8_t i = 0; i < NUM_PIR; i++) {
    bool state = digitalRead(pirPins[i]) == HIGH;
    if (state && !lastState[i]) {
      motionStart[i] = now;
      lastTrigger[i] = now;
      triggerCount[i]++;
      info("[%s] PIR %d (GPIO %d) -> motion  (#%lu)", stamp(now), i + 1, pirPins[i],
           (unsigned long)triggerCount[i]);
    } else if (!state && lastState[i]) {
      info("[%s] PIR %d (GPIO %d) -> clear   (%.1f s)", stamp(now), i + 1, pirPins[i],
           (now - motionStart[i]) / 1000.0);
    }
    lastState[i] = state;
    any |= state;
  }
  digitalWrite(LED_PIN, any);

  if (any && !wasAny) info("[%s] area occupied", stamp(now));
  if (!any && wasAny) info("[%s] area clear", stamp(now));

  if (any && (!wasAny || now - lastMotionMsg >= REPEAT_MS)) {
    Serial.println("MOTION");
    lastMotionMsg = now;
  }
  wasAny = any;

  if (now - lastBeat >= HEARTBEAT_MS) {
    Serial.println("ALIVE");
    lastBeat = now;
  }

  if (showStatus && now - lastStatus >= STATUS_MS) {
    printStatus(now);
    lastStatus = now;
  }

  delay(50);
}