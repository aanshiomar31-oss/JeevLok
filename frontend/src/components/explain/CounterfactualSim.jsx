import { useState, useEffect } from "react";
import { evaluateCounterfactual } from "../../services/api.js";
import PriorityBadge from "../PriorityBadge.jsx";

export default function CounterfactualSim({ basePatient = null }) {
  const [sbp, setSbp] = useState(basePatient?.sbp || 110);
  const [hr, setHr] = useState(basePatient?.heartrate || 80);
  const [o2sat, setO2sat] = useState(basePatient?.o2sat || 98);
  const [resprate, setResprate] = useState(basePatient?.resprate || 18);

  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (basePatient) {
      if (basePatient.sbp) setSbp(basePatient.sbp);
      if (basePatient.heartrate) setHr(basePatient.heartrate);
      if (basePatient.o2sat) setO2sat(basePatient.o2sat);
      if (basePatient.resprate) setResprate(basePatient.resprate);
    }
  }, [basePatient]);

  const runSimulation = async () => {
    const patient = basePatient || {
      age: 55,
      gender: "M",
      sbp: 90,
      dbp: 60,
      heartrate: 115,
      resprate: 24,
      o2sat: 89,
      chest_pain: true,
      diaphoresis: true,
    };

    setLoading(true);
    try {
      const data = await evaluateCounterfactual(patient, {
        sbp: Number(sbp),
        heartrate: Number(hr),
        o2sat: Number(o2sat),
        resprate: Number(resprate),
      });
      setComparison(data);
    } catch (err) {
      console.warn("Counterfactual simulation error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runSimulation();
  }, [sbp, hr, o2sat, resprate]);

  return (
    <div className="panel p-6 border border-surface-border shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-surface-border pb-4">
        <div>
          <h2 className="text-base font-bold text-surface-ink flex items-center gap-2">
            <span>🔬</span> "What-If" Counterfactual Triage Simulator
          </h2>
          <p className="text-xs text-surface-muted">
            Simulate how physiological stabilization or decompensation dynamically shifts AI risk score and priority.
          </p>
        </div>
        <button
          type="button"
          onClick={runSimulation}
          disabled={loading}
          className="btn-secondary text-xs px-3 py-1.5"
        >
          {loading ? "Calculating…" : "Re-simulate"}
        </button>
      </div>

      {/* Sliders Grid */}
      <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* SBP */}
        <div className="rounded-xl border border-surface-border bg-slate-50/50 p-3 dark:bg-slate-900/40">
          <div className="flex items-center justify-between text-xs font-medium">
            <span className="text-surface-muted">Systolic BP</span>
            <span className="font-bold text-surface-ink">{sbp} mmHg</span>
          </div>
          <input
            type="range"
            min="60"
            max="200"
            step="1"
            value={sbp}
            onChange={(e) => setSbp(e.target.value)}
            className="mt-2 w-full accent-indigo-600 cursor-pointer"
          />
          <div className="mt-1 flex justify-between text-[10px] text-slate-400">
            <span>60 (Shock)</span>
            <span>120 (Norm)</span>
            <span>200 (HTN)</span>
          </div>
        </div>

        {/* Heart Rate */}
        <div className="rounded-xl border border-surface-border bg-slate-50/50 p-3 dark:bg-slate-900/40">
          <div className="flex items-center justify-between text-xs font-medium">
            <span className="text-surface-muted">Heart Rate</span>
            <span className="font-bold text-surface-ink">{hr} bpm</span>
          </div>
          <input
            type="range"
            min="40"
            max="180"
            step="1"
            value={hr}
            onChange={(e) => setHr(e.target.value)}
            className="mt-2 w-full accent-indigo-600 cursor-pointer"
          />
          <div className="mt-1 flex justify-between text-[10px] text-slate-400">
            <span>40 (Brady)</span>
            <span>75 (Norm)</span>
            <span>180 (Tachy)</span>
          </div>
        </div>

        {/* O2 Sat */}
        <div className="rounded-xl border border-surface-border bg-slate-50/50 p-3 dark:bg-slate-900/40">
          <div className="flex items-center justify-between text-xs font-medium">
            <span className="text-surface-muted">SpO2</span>
            <span className="font-bold text-surface-ink">{o2sat}%</span>
          </div>
          <input
            type="range"
            min="70"
            max="100"
            step="1"
            value={o2sat}
            onChange={(e) => setO2sat(e.target.value)}
            className="mt-2 w-full accent-indigo-600 cursor-pointer"
          />
          <div className="mt-1 flex justify-between text-[10px] text-slate-400">
            <span>70% (Hypoxia)</span>
            <span>95% (Safe)</span>
            <span>100%</span>
          </div>
        </div>

        {/* Resp Rate */}
        <div className="rounded-xl border border-surface-border bg-slate-50/50 p-3 dark:bg-slate-900/40">
          <div className="flex items-center justify-between text-xs font-medium">
            <span className="text-surface-muted">Resp Rate</span>
            <span className="font-bold text-surface-ink">{resprate} /min</span>
          </div>
          <input
            type="range"
            min="8"
            max="40"
            step="1"
            value={resprate}
            onChange={(e) => setResprate(e.target.value)}
            className="mt-2 w-full accent-indigo-600 cursor-pointer"
          />
          <div className="mt-1 flex justify-between text-[10px] text-slate-400">
            <span>8 (Bradypnea)</span>
            <span>16 (Norm)</span>
            <span>40 (Tachypnea)</span>
          </div>
        </div>
      </div>

      {/* Comparison Results */}
      {comparison && (
        <div className="mt-6 space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {/* Baseline Card */}
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-800/40">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Baseline Prediction
              </span>
              <div className="mt-2 flex items-center justify-between">
                <PriorityBadge priority={comparison.original?.priority} />
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Risk Score: <span className="font-bold text-surface-ink">{comparison.original?.risk_score}</span>/100
                </span>
              </div>
            </div>

            {/* Counterfactual Card */}
            <div className="rounded-xl border border-indigo-200 bg-gradient-to-br from-indigo-50/40 to-purple-50/40 p-4 shadow-sm dark:border-indigo-900 dark:from-slate-800 dark:to-indigo-950/30">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-700 dark:text-indigo-400">
                  Counterfactual Trajectory
                </span>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                    comparison.risk_delta < 0
                      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                      : comparison.risk_delta > 0
                      ? "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300"
                      : "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300"
                  }`}
                >
                  {comparison.risk_delta > 0 ? `+${comparison.risk_delta}` : comparison.risk_delta} pts
                </span>
              </div>

              <div className="mt-2 flex items-center justify-between">
                <PriorityBadge priority={comparison.counterfactual?.priority} />
                <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Simulated Risk: <span className="font-bold text-surface-ink">{comparison.counterfactual?.risk_score}</span>/100
                </span>
              </div>
            </div>
          </div>

          {/* Natural Language Explanation of What Changed */}
          <div className="rounded-xl bg-slate-100/80 p-3.5 text-xs leading-relaxed text-slate-800 dark:bg-slate-800/60 dark:text-slate-200">
            <span className="font-bold text-indigo-700 dark:text-indigo-400">Impact Analysis: </span>
            {comparison.natural_language_explanation}
          </div>

          {/* Feature Impact Breakdown */}
          {comparison.feature_impact_cards?.length > 0 && (
            <div className="space-y-2 pt-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                Active Contributing Factors
              </span>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {comparison.feature_impact_cards.map((card, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between rounded-lg border border-surface-border bg-white p-2.5 text-xs dark:bg-slate-800/40"
                  >
                    <div>
                      <span className="font-semibold text-surface-ink">{card.feature_name}</span>
                      <p className="text-[11px] text-surface-muted">{card.clinical_note}</p>
                    </div>
                    <span
                      className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase ${
                        card.direction === "escalating"
                          ? "bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300"
                          : "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300"
                      }`}
                    >
                      {card.direction}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
