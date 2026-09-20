export default function EntityChips({
  urgencyLevel = "LOW",
  urgencyKeywords = [],
  entityChips = [],
  icdMappings = [],
}) {
  if (!entityChips.length && !icdMappings.length && !urgencyKeywords.length) {
    return null;
  }

  const urgencyStyles = {
    CRITICAL: "bg-red-500/15 text-red-700 border-red-300 dark:text-red-400 dark:border-red-800",
    HIGH: "bg-amber-500/15 text-amber-800 border-amber-300 dark:text-amber-400 dark:border-amber-800",
    MODERATE: "bg-yellow-500/15 text-yellow-800 border-yellow-300 dark:text-yellow-400 dark:border-yellow-800",
    LOW: "bg-emerald-500/15 text-emerald-800 border-emerald-300 dark:text-emerald-400 dark:border-emerald-800",
  };

  return (
    <div className="space-y-3 rounded-xl border border-surface-border bg-slate-50/60 p-3.5 dark:bg-slate-900/40">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-surface-border/60 pb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          NLP Clinical Entity Extraction
        </span>
        {urgencyLevel && (
          <span
            className={`rounded-full border px-2.5 py-0.5 text-[11px] font-bold tracking-wide ${
              urgencyStyles[urgencyLevel] || urgencyStyles.LOW
            }`}
          >
            {urgencyLevel} URGENCY
          </span>
        )}
      </div>

      {/* Extracted Entity Chips */}
      <div className="flex flex-wrap gap-1.5">
        {entityChips.map((chip, idx) => {
          let chipClass = "bg-slate-200/80 text-slate-800 border-slate-300";
          let icon = "•";

          if (chip.type === "vital") {
            chipClass = "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 dark:border-blue-800";
            icon = "📊";
          } else if (chip.is_negated) {
            chipClass = "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800 line-through opacity-85";
            icon = "✕";
          } else if (chip.type === "finding") {
            chipClass = "bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-800 font-semibold";
            icon = "⚠";
          }

          return (
            <span
              key={idx}
              className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs ${chipClass}`}
            >
              <span className="text-[10px]">{icon}</span>
              {chip.label}
            </span>
          );
        })}
      </div>

      {/* ICD-10 Mappings */}
      {icdMappings.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          <span className="text-[11px] font-medium text-slate-500">ICD-10:</span>
          {icdMappings.map((m, idx) => (
            <span
              key={idx}
              title={`${m.standard_name}: ${m.icd10_desc}`}
              className="inline-flex items-center gap-1 rounded-md border border-purple-200 bg-purple-50 px-2 py-0.5 text-[11px] font-medium text-purple-700 dark:border-purple-800 dark:bg-purple-950/40 dark:text-purple-300"
            >
              <span className="font-mono font-bold">{m.icd10}</span>
              <span>{m.matched_term}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
