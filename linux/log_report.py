"""Resumen del log persistente (logs/stt-*.jsonl, ver log.py) para ajustar
el .env con datos y no a ojo.

    python log_report.py              # últimos 7 días
    python log_report.py --days 1     # último día
    python log_report.py --run last   # sólo la última corrida del servicio
    python log_report.py --dir /otra/carpeta

Qué muestra:
- Corridas (arranques), qué parámetros cambiaron entre una y otra, cómo
  terminó cada una (cierre limpio o cortada: apagón/desenchufe) y los cambios
  de código (changelog.py) que aparecen por primera vez en el rango.
- VAD: frases cerradas por motivo; nivel de las aceptadas vs las descartadas
  por lejanas; cuántas quedaron "casi" (a <20 % del umbral); ráfagas de voz
  que no llegaron a abrir.
- Ruido de fondo (heartbeat): percentiles y cuánto tiempo estuvo en el tope
  NOISE_FLOOR_MAX (si es mucho, el umbral real lo pone el ruido).
- STT: destino de cada frase, latencia y confianza de Groq, métricas de
  Whisper (logprob/no_speech) de lo enviado vs lo descartado, y las frases
  enviadas con confianza baja (candidatas a basura).
- Turnos del robot (speak): latencia desde el envío hasta que habla, duración,
  SPEAK_END perdidos; órdenes de corte (barge) y si el robot se calló.
- Eventos al orquestador (orch_ev): cuántos no llegaron.
- Voz (speaker_id): similitudes aceptadas / rechazadas, rechazos cerca del
  umbral, reconocimientos al despertar, decisiones de la memoria de voces.
- Frases mezcladas (mix_trim.py, MIX_TRIM_ENABLED): cuántas se recortaron,
  por qué no las otras, y cuánto tardó el cómputo en la Pi.
- Pistas: reglas simples sobre los números de arriba. Son sugerencias para
  mirar, no ajustes automáticos.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter

import numpy as np

ROOT_LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def load(directory: str, days: float) -> list[dict]:
    since = time.time() - days * 86400
    records = []
    for path in sorted(glob.glob(os.path.join(directory, "stt-*.jsonl"))):
        if os.path.getmtime(path) < since:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # línea cortada por un apagón
                if rec.get("t", 0) >= since:
                    records.append(rec)
    return records


def pct(values, *qs) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return " ".join(f"p{q}={np.percentile(values, q):.4g}" for q in qs) + f" (n={len(values)})"


def events(records, name):
    return [r for r in records if r.get("k") == "ev" and r.get("ev") == name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", default=os.getenv("LOG_DIR", "") or ROOT_LOGS)
    parser.add_argument("--days", type=float, default=7)
    parser.add_argument("--run", help="'last' o un id de corrida")
    args = parser.parse_args()

    records = load(args.dir, args.days)
    if not records:
        print(f"Sin registros en {os.path.abspath(args.dir)} (últimos {args.days:g} días).")
        return
    configs = events(records, "config")
    if args.run:
        run = configs[-1]["run"] if args.run == "last" and configs else args.run
        records = [r for r in records if r.get("run") == run]
        configs = events(records, "config")

    t0, t1 = records[0]["t"], records[-1]["t"]
    fmt_t = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    print(f"== Log STT: {fmt_t(t0)} -> {fmt_t(t1)}  ({len(records)} registros, "
          f"{len(configs)} arranques)\n")

    # --- corridas y cambios de config ---------------------------------------
    cfg = configs[-1] if configs else {}
    prev = None
    for c in configs:
        line = f"  {fmt_t(c['t'])} run={c['run']} git={c.get('git')} mic={c.get('MIC_MODE')}"
        if prev is not None:
            skip = {"t", "run", "k", "ev", "voices", "changes", "boot_uptime_s"}
            diff = [f"{k}: {prev.get(k)} -> {c.get(k)}" for k in sorted(set(c) | set(prev))
                    if k not in skip and prev.get(k) != c.get(k)]
            if diff:
                line += "\n      cambió: " + "; ".join(diff[:12])
        run_recs = [r for r in records if r.get("run") == c["run"]]
        ends = [r for r in run_recs if r.get("ev") == "exit"]
        if ends:
            line += f"\n      cerró: {ends[-1].get('reason')} tras {ends[-1].get('ran_s', 0) / 60:.0f} min"
        elif c is not configs[-1]:
            line += (f"\n      SIN cierre: último registro {fmt_t(run_recs[-1]['t'])} "
                     "(apagón, desenchufe, kill -9 o cuelgue)")
        if c.get("boot_uptime_s") is not None and c["boot_uptime_s"] < 600:
            line += f" [la Pi recién arrancaba: {c['boot_uptime_s']:.0f}s]"
        print(line)
        prev = c
    seen_changes: set = set()
    new_changes = []
    for c in configs:
        for ch in c.get("changes") or []:
            if ch.get("id") not in seen_changes:
                seen_changes.add(ch.get("id"))
                new_changes.append((c, ch))
    if new_changes:
        print("  cambios de código vigentes desde (changelog.py):")
        for c, ch in new_changes:
            print(f"    [{ch.get('date')}] {ch.get('id')} (desde run {c['run']}): {ch.get('what')}")
    elif configs and not any(c.get("changes") for c in configs):
        print("  (corridas anteriores al changelog: sin cambios registrados en el log)")
    if cfg:
        keys = ("NEAR_RMS_THRESHOLD", "RMS_THRESHOLD", "NEAR_SNR_RATIO", "NOISE_FLOOR_MAX",
                "ATTENTION_LEVEL_RATIO", "SPEAKER_MIN_SIMILARITY_NEW",
                "SPEAKER_MIN_SIMILARITY_KNOWN", "SPEAKER_KNOWN_S", "SPEAKER_MEMORY",
                "SILERO_THRESHOLD", "WAKE_WINDOW_S")
        print("  config actual: " + ", ".join(f"{k}={cfg.get(k)}" for k in keys if k in cfg))
    print()

    # --- VAD ------------------------------------------------------------------
    utts = events(records, "utt")
    weak = events(records, "weak")
    reasons = Counter(u.get("reason") for u in utts)
    print("== VAD (frases cerradas)")
    print("  por motivo: " + ", ".join(f"{k}={n}" for k, n in reasons.most_common()))
    ok_levels = [u["level"] for u in utts if u.get("accepted")]
    far = [u for u in utts if u.get("reason") == "lejana"]
    almost = [u for u in far if u.get("near") and u["level"] >= 0.8 * u["near"]]
    print(f"  nivel aceptadas:   {pct(ok_levels, 10, 50, 90)}")
    print(f"  nivel lejanas:     {pct([u['level'] for u in far], 10, 50, 90)}")
    print(f"  umbral cerca usado:{pct([u.get('near') for u in utts], 10, 50, 90)}")
    print(f"  lejanas 'casi' (>=80 % del umbral): {len(almost)} de {len(far)}")
    print(f"  ráfagas que no abrieron: {len(weak)}; rms_max {pct([w['rms_max'] for w in weak], 50, 90)}"
          f" vs abre {pct([w['open'] for w in weak], 50)}")
    print()

    # --- ruido ----------------------------------------------------------------
    hbs = events(records, "hb")
    noise = [h["noise"] for h in hbs]
    cap = cfg.get("NOISE_FLOOR_MAX")
    print("== Ruido de fondo (heartbeat)")
    print(f"  piso:       {pct(noise, 10, 50, 90)}")
    print(f"  abre:       {pct([h['open'] for h in hbs], 10, 50, 90)}")
    print(f"  cerca:      {pct([h['near'] for h in hbs], 10, 50, 90)}")
    at_cap = None
    if cap and noise:
        at_cap = float(np.mean([n >= 0.95 * cap for n in noise]))
        print(f"  en el tope NOISE_FLOOR_MAX={cap}: {at_cap:.0%} del tiempo")
    awake = [h for h in hbs if h.get("awake")]
    print(f"  tiempo despierto: {len(awake) * (cfg.get('heartbeat_s') or 10) / 60:.1f} min; "
          f"muteado (robot hablando): {sum(h.get('muted_frames', 0) for h in hbs) * 0.03 / 60:.1f} min")
    print()

    # --- STT --------------------------------------------------------------------
    stt = events(records, "stt")
    outcomes = Counter(s.get("outcome") for s in stt)
    print("== STT (destino de cada frase)")
    print("  " + ", ".join(f"{k}={n}" for k, n in outcomes.most_common()))
    print(f"  latencia Groq: {pct([s.get('stt_s') for s in stt], 50, 90, 99)}")
    print(f"  confianza:     {pct([s.get('conf') for s in stt if s.get('text')], 10, 50)}")
    print(f"  audio enviado: {pct([s.get('audio_s') for s in stt if 'stt_s' in s], 50, 90)} s")
    sent = [s for s in stt if s.get("outcome") == "enviada"]
    with_m = [s for s in stt if s.get("logprob_min") is not None]
    if with_m:
        print(f"  logprob_min enviadas:    {pct([s['logprob_min'] for s in sent if s.get('logprob_min') is not None], 10, 50)}")
        print(f"  logprob_min descartadas: {pct([s['logprob_min'] for s in with_m if s.get('outcome') != 'enviada'], 10, 50)}")
        print(f"  no_speech_max enviadas:  {pct([s.get('no_speech_max') for s in sent], 50, 90)}")
        filtered = [s for s in with_m if s.get("raw_text")]
        for s in filtered[-5:]:
            print(f"    filtrado: «{s['raw_text']}» -> «{s.get('text', '')}»")
    low = [s for s in sent if s.get("conf") is not None and s["conf"] < 0.45]
    if low:
        print(f"  enviadas con confianza < 0.45: {len(low)} de {len(sent)}")
        for s in low[-8:]:
            print(f"    conf={s['conf']:.2f} logprob_min={s.get('logprob_min')} «{s.get('sent_text') or s.get('text')}»")
    print()

    # --- turnos del robot / cortes / eventos ------------------------------------
    speaks = events(records, "speak")
    barges = events(records, "barge")
    orch = events(records, "orch_ev")
    timeouts = [e for e in speaks if e.get("end") == "timeout"]
    stopped = [e for e in speaks if e.get("stops")]
    if speaks or barges or orch:
        print("== Robot hablando (speak / barge / orch_ev)")
        for kind in ("texto", "ack"):
            rs = [e.get("reply_s") for e in speaks if e.get("reply") == kind]
            if rs:
                label = "transcript -> habla" if kind == "texto" else "wake -> «Sí, dime»"
                print(f"  latencia {label}: {pct(rs, 50, 90)} s")
        print(f"  turnos: {len(speaks)}; duración {pct([e.get('dur_s') for e in speaks], 50, 90)} s; "
              f"SPEAK_END perdidos: {len(timeouts)}")
        if barges:
            print("  cortes oídos: " + ", ".join(f"{k}={n}" for k, n in Counter(
                b.get("outcome") for b in barges).most_common())
                  + f"; re-despertó {sum(1 for b in barges if b.get('rewake'))}"
                  f" de {sum(1 for b in barges if b.get('rewake') is not None)}")
        if stopped:
            print(f"  tras un corte siguió hablando: {pct([e.get('after_stop_s') for e in stopped], 50, 90)} s")
        failed = Counter(e.get("name") for e in orch if not e.get("ok"))
        print(f"  eventos al orquestador: {len(orch)}; no llegaron: "
              + (", ".join(f"{k}={n}" for k, n in failed.most_common()) or "0"))
        print()

    # --- voz -------------------------------------------------------------------
    judged = [s for s in stt if s.get("spk_sim") is not None]
    # Umbral dinámico: el de cada frase (spk_th); logs viejos, el fijo de entonces.
    th_fixed = cfg.get("SPEAKER_MIN_SIMILARITY", 0.35)

    def th_of(s: dict) -> float:
        return s.get("spk_th") if s.get("spk_th") is not None else th_fixed

    acc = [s["spk_sim"] for s in judged if s.get("spk_ok")]
    rej = [s["spk_sim"] for s in judged if not s.get("spk_ok")]
    near_rej = [s for s in judged if not s.get("spk_ok") and s["spk_sim"] >= th_of(s) - 0.1]
    ths = [s.get("spk_th") for s in judged if s.get("spk_th") is not None]
    print("== Voz (speaker_id, umbral " + (f"dinámico {pct(ths, 5, 50, 95)}" if ths
                                           else str(th_fixed)) + ")")
    print(f"  aceptadas:  {pct(acc, 5, 50)}")
    print(f"  rechazadas: {pct(rej, 50, 95)}")
    print(f"  rechazos a <0.1 del umbral: {len(near_rej)}"
          + "".join(f"\n    sim={s['spk_sim']:.2f} (umbral {th_of(s):.2f}) "
                  f"[{s.get('voice') or 'voz nueva'}] «{s.get('text', '')}»" for s in near_rej[-5:]))
    rec = events(records, "spk_recognize")
    learned = Counter(e.get("decision") for e in events(records, "spk_learn"))
    print(f"  al despertar: {sum(1 for e in rec if e.get('voice'))} reconocidas de {len(rec)}; "
          f"best_sim {pct([e.get('best_sim') for e in rec], 50, 90)}")
    print("  memoria: " + (", ".join(f"{k}={n}" for k, n in learned.most_common()) or "-"))
    wakes = events(records, "wake")
    print(f"  wakes: {sum(1 for w in wakes if w.get('accepted'))} aceptados, "
          f"{sum(1 for w in wakes if not w.get('accepted'))} rechazados")
    print()

    # --- frases mezcladas (mix_trim.py) ------------------------------------------
    mixes = [s for s in stt if s.get("mix")]
    if mixes:
        tried = [s for s in mixes if s["mix"].get("result") not in (
            "sin datos por tramo", "ningún tramo es claramente de otro",
            "ningún tramo es de quien llamó")]
        cut_ok = [s for s in tried if s["mix"].get("result") == "recortada"]
        print(f"== Frases mezcladas (MIX_TRIM, {len(mixes)} juzgadas por tramos)")
        print(f"  con mezcla: {len(tried)}; recortadas y enviadas: {len(cut_ok)}")
        print("  por qué no: " + (", ".join(f"{k}={n}" for k, n in Counter(
            s["mix"].get("result") for s in mixes if s["mix"].get("result") != "recortada"
        ).most_common()) or "-"))
        print(f"  cómputo: {pct([s['mix'].get('compute_s') for s in mixes], 50, 90)} s")
        for s in cut_ok[-5:]:
            m = s["mix"]
            print(f"    {m.get('uso')}  {m.get('kept_s')}/{m.get('total_s')} s  «{s.get('text', '')}»")
        print()

    # --- pistas ----------------------------------------------------------------
    hints = []
    if far and len(almost) >= max(3, 0.3 * len(far)):
        hints.append(f"{len(almost)} frases descartadas por lejanas quedaron a <20 % del umbral: "
                     "si eran tuyas, bajar NEAR_RMS_THRESHOLD / NEAR_SNR_RATIO un poco.")
    if at_cap is not None and at_cap > 0.5:
        hints.append(f"El piso de ruido estuvo en el tope {at_cap:.0%} del tiempo: el umbral real lo "
                     "pone ruido x NEAR_SNR_RATIO, no NEAR_RMS_THRESHOLD. Revisar ganancia/ruido del "
                     "mic o bajar NEAR_SNR_RATIO.")
    if len(weak) > max(5, len(utts)):
        hints.append("Hay más ráfagas de voz que no abrieron que frases cerradas: el umbral de "
                     "apertura (RMS_THRESHOLD / ruido x SNR) puede estar alto.")
    if judged and len(near_rej) >= max(2, 0.2 * len(rej)):
        hints.append("Varios rechazos de voz cerca del umbral: mirar los textos de arriba; si eran "
                     "tuyos, bajar SPEAKER_MIN_SIMILARITY_NEW o _KNOWN (según la voz) 0.05.")
    if outcomes.get("vacia", 0) > 0.2 * max(1, sum(outcomes.values())):
        hints.append("Más del 20 % de las frases vuelven vacías de Groq: suele ser ruido que pasó el "
                     "VAD (subir SILERO_THRESHOLD o el umbral de nivel).")
    stt_s = [s.get("stt_s") for s in stt if s.get("stt_s") is not None]
    if stt_s and np.percentile(stt_s, 90) > 2.0:
        hints.append("La latencia de Groq p90 pasa 2 s: revisar red/WiFi de la Pi.")
    if timeouts:
        hints.append(f"{len(timeouts)} turnos del robot sin SPEAK_END (desmute por timeout): "
                     "revisar en el log del orquestador si se colgó/reinició o si no lo manda.")
    slow_stop = [e for e in stopped if (e.get("after_stop_s") or 0) > 3]
    if slow_stop:
        hints.append(f"{len(slow_stop)} veces el robot siguió hablando >3 s después de una orden de "
                     "corte: el orquestador no está cortando el TTS al recibir @@event:stop.")
    if orch and sum(1 for e in orch if not e.get("ok")) >= 2:
        hints.append("Eventos al orquestador que no llegaron (ver arriba): si se repite, poner "
                     "LAN y Tailscale en ORCHESTRATOR_IP.")
    unclean = [c for c in configs[:-1]
               if not any(r.get("ev") == "exit" and r.get("run") == c["run"] for r in records)]
    if unclean:
        hints.append(f"{len(unclean)} corrida(s) terminaron sin cierre: si nadie la desenchufó, "
                     "mirar alimentación (POWER) y activar journal persistente en la Pi.")
    print("== Pistas")
    print("\n".join(f"  - {h}" for h in hints) if hints else "  (nada llamativo)")


if __name__ == "__main__":
    main()
