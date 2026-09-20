import { useState, useEffect } from "react";
import { generateClinicalSummary } from "../../services/api.js";

export default function ClinicalSummaryCard({ patientData, triageResult }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!patientData || !triageResult) return;

    let isMounted = true;
    setLoading(true);
    generateClinicalSummary(patientData, triageResult)
      .then((data) => {
        if (isMounted) setSummary(data);
      })
      .catch((err) => {
        console.warn("Summary generation failed:", err);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [JSON.stringify(patientData), JSON.stringify(triageResult)]);

  if (loading) {
    return (
      <div className="rounded-xl border border-surface-border bg-surface p-5 text-center shadow-sm">
        <div className="inline-flex items-center gap-2 text-xs font-semibold text-accent-mintInk">
          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent-mint border-t-transparent" />
          Synthesizing Explainable Clinical Summary…
        </div>
      </div>
    );
  }

  if (!summary) return null;

  const vitals = summary.vitals_summary || {};
  const shockIndex = vitals.shock_index;
  const mapVal = vitals.map;

  return (
    <div className="rounded-xl border border-indigo-200 bg-gradient-to-br from-indigo-50/40 via-white to-slate-50/60 p-5 shadow-sm dark:border-indigo-950 dark:from-slate-900 dark:via-slate-900/90 dark:to-indigo-950/20">
      <div className="flex items-center justify-between border-b border-indigo-100 pb-3 dark:border-indigo-900/50">
        <div className="flex items-center gap-2">
          <span className="text-base">📋</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-indigo-900 dark:text-indigo-300">
            Explainable Clinical Summary
          </h3>
        </div>
        <span className="rounded-full bg-indigo-100 px-2.5 py-0.5 text-[11px] font-bold text-indigo-800 dark:bg-indigo-950 dark:text-indigo-300">
          {summary.priority} • Risk {summary.risk_score}/100
        </span>
      </div>

      {/* Synthesis */}
      <p className="mt-3 text-xs leading-relaxed text-slate-700 dark:text-slate-300">
        {summary.clinical_synthesis}
      </p>

      {/* Physiological Indicators: Shock Index & MAP */}
      {(shockIndex !== null || mapVal !== null) && (
        <div className="mt-3 grid grid-cols-2 gap-2">
          {shockIndex !== null && (
            <div
              className={`rounded-lg border p-2 text-xs ${
                shockIndex >= 0.9
                  ? "border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300"
                  : "border-slate-200 bg-white text-slate-700 dark:border-slate-800 dark:bg-slate-800/60 dark:text-slate-300"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold">Shock Index: {shockIndex}</span>
                <span className="text-[10px] uppercase font-bold">
                  {shockIndex >= 0.9 ? "⚠ Elevated" : "Normal"}
                </span>
              </div>
              <p className="mt-0.5 text-[11px] opacity-85">
                HR {vitals.heartrate} / SBP {vitals.sbp} (Norm: 0.5–0.7)
              </p>
            </div>
          )}

          {mapVal !== null && (
            <div
              className={`rounded-lg border p-2 text-xs ${
                mapVal < 65
                  ? "border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300"
                  : "border-slate-200 bg-white text-slate-700 dark:border-slate-800 dark:bg-slate-800/60 dark:text-slate-300"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold">MAP: {mapVal} mmHg</span>
                <span className="text-[10px] uppercase font-bold">
                  {mapVal < 65 ? "⚠ Low" : "Adequate"}
                </span>
              </div>
              <p className="mt-0.5 text-[11px] opacity-85">
                Target: &gt;65 mmHg for perfusion
              </p>
            </div>
          )}
        </div>
      )}

      {/* Urgency Drivers */}
      {summary.urgency_drivers?.length > 0 && (
        <div className="mt-3">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
            Key Contributing Rationale:
          </span>
          <ul className="mt-1 space-y-1 text-xs text-slate-700 dark:text-slate-300">
            {summary.urgency_drivers.map((driver, idx) => (
              <li key={idx} className="flex items-start gap-1.5">
                <span className="text-red-500 font-bold">•</span>
                <span>{driver}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Suggested Protocols */}
      {summary.suggested_protocols?.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5 pt-2 border-t border-indigo-100/70 dark:border-indigo-900/40">
          <span className="text-[11px] font-semibold text-slate-500">Suggested Protocols:</span>
          {summary.suggested_protocols.map((proto, idx) => (
            <span
              key={idx}
              className="rounded-full border border-blue-200 bg-blue-50 px-2.5 py-0.5 text-[11px] font-semibold text-blue-800 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-300"
            >
              🩺 {proto}
            </span>
          ))}
        </div>
      )}

      {/* Clinical Disclaimer */}
      <div className="mt-3 border-t border-indigo-100/80 pt-2 text-center dark:border-indigo-900/40">
        <p className="text-[11px] font-bold text-slate-500 italic">
          "{summary.disclaimer}"
        </p>
      </div>
    </div>
  );
}
