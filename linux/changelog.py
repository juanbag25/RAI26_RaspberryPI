"""Cambios de comportamiento que importan para leer el log persistente.

Cada arranque escribe las entradas de acá en el evento `config` (campo
`changes`), y log_report.py muestra cuáles aparecen por primera vez en el
rango del reporte. Así, mirando un log, se sabe desde cuándo rige cada
cambio sin cruzarlo a mano con el git log (el commit también va: `git`).

Agregar una entrada nueva ARRIBA cuando un cambio altere lo que se ve en el
log (un descarte nuevo, un evento nuevo, un umbral con otro significado).
`id` corto y estable: es lo que se compara entre corridas.
"""

CHANGES: list[dict] = [
    {
        "id": "rewake-barge-level",
        "date": "2026-10-16",
        "what": "«oye rai» como orden de corte mientras el robot habla ahora "
                "re-despierta: el traspaso usa el pico del barge-in como nivel "
                "(antes el VAD muteado daba 0 y se rechazaba siempre con "
                "traspaso_lejano, con MIC_MODE=normal).",
    },
    {
        "id": "prompt-sin-vocabulario",
        "date": "2026-10-16",
        "what": "STT_PROMPT sin el rótulo «Vocabulario:» (Whisper lo devolvía como "
                "texto: «¡Vocabulario!») y «*vocabulario» en STT_HALLUCINATIONS.",
    },
    {
        "id": "log-turnos-cortes",
        "date": "2026-10-16",
        "what": "Eventos nuevos: speak (cada turno del robot: dur_s, starts, end "
                "end|timeout, reply/reply_s = latencia desde el envío, stops, "
                "after_stop_s), barge (cada orden de corte oída: outcome "
                "corte|flojo|parlante, rewake), orch_ev (cada evento al "
                "orquestador y si llegó), exit / prev_run (cierre limpio o no), "
                "y en stt las métricas de Whisper (segs, segs_drop, logprob_min, "
                "no_speech_max, compression_max, raw_text).",
    },
]
