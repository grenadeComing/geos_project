import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";

const data = [
  {
    model: "Claude Opus 4.6",
    "One-shot": 73.5,
    "CoT": 71.5,
    "Agent (TOC)": 89.0,
  },
  {
    model: "Claude Sonnet 4.6",
    "One-shot": 65.0,
    "CoT": 70.0,
    "Agent (TOC)": null,
  },
  {
    model: "GPT-5.2",
    "One-shot": 55.0,
    "CoT": 61.5,
    "Agent (TOC)": 73.5,
  },
  {
    model: "GPT-5.1",
    "One-shot": 63.5,
    "CoT": 61.0,
    "Agent (TOC)": 69.0,
  },
  {
    model: "Gemini 2.5 Flash",
    "One-shot": 40.0,
    "CoT": null,
    "Agent (TOC)": 33.0,
  },
  {
    model: "Gemini 2.5 Pro",
    "One-shot": 63.0,
    "CoT": null,
    "Agent (TOC)": null,
  },
];

const COLORS = {
  "One-shot": "#94a3b8",
  "CoT": "#60a5fa",
  "Agent (TOC)": "#2563eb",
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload) return null;
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-lg">
      <p className="font-semibold text-gray-800 mb-1">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color }} className="text-sm">
          {entry.name}: {entry.value !== null ? `${entry.value}%` : "—"}
        </p>
      ))}
    </div>
  );
};

const ablationData = [
  {
    model: "Claude Opus 4.6",
    "Agent (TOC)": 89.0,
    "Agent (Raw)": 91.5,
  },
  {
    model: "GPT-5.2",
    "Agent (TOC)": 73.5,
    "Agent (Raw)": 72.5,
  },
  {
    model: "GPT-5.1",
    "Agent (TOC)": 69.0,
    "Agent (Raw)": 71.5,
  },
];

const tokenData = [
  {
    model: "GPT-5.1",
    "TOC": 7.03,
    "Raw": 8.96,
  },
  {
    model: "GPT-5.2",
    "TOC": 8.90,
    "Raw": 11.62,
  },
];

const ABLATION_COLORS = {
  "Agent (TOC)": "#2563eb",
  "Agent (Raw)": "#f59e0b",
};

const TOKEN_COLORS = {
  "TOC": "#2563eb",
  "Raw": "#f59e0b",
};

export default function EvaluationCharts() {
  const [activeChart, setActiveChart] = useState("main");

  return (
    <div className="bg-white min-h-screen p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          PHREEQC-Agent Evaluation Results
        </h1>
        <p className="text-gray-500 mb-6">Paper #322 — ACM CAIS 2026</p>

        <div className="flex gap-2 mb-6">
          {[
            { id: "main", label: "Accuracy Comparison" },
            { id: "ablation", label: "TOC vs Raw" },
            { id: "tokens", label: "Token Usage" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveChart(tab.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeChart === tab.id
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeChart === "main" && (
          <div>
            <h2 className="text-lg font-semibold text-gray-800 mb-1">
              Accuracy by Method
            </h2>
            <p className="text-sm text-gray-500 mb-4">
              One-shot (no reasoning) vs CoT (reasoning, no tools) vs Agent (tools + reasoning)
            </p>
            <ResponsiveContainer width="100%" height={420}>
              <BarChart
                data={data}
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                barCategoryGap="20%"
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="model"
                  tick={{ fontSize: 12, fill: "#374151" }}
                  angle={-15}
                  textAnchor="end"
                  height={60}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: 12, fill: "#374151" }}
                  label={{
                    value: "Accuracy (%)",
                    angle: -90,
                    position: "insideLeft",
                    style: { fontSize: 13, fill: "#374151" },
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  wrapperStyle={{ fontSize: 13 }}
                  iconType="square"
                />
                <Bar dataKey="One-shot" fill={COLORS["One-shot"]} radius={[3, 3, 0, 0]}>
                  <LabelList
                    dataKey="One-shot"
                    position="top"
                    formatter={(v) => (v ? `${v}%` : "")}
                    style={{ fontSize: 10, fill: "#6b7280" }}
                  />
                </Bar>
                <Bar dataKey="CoT" fill={COLORS["CoT"]} radius={[3, 3, 0, 0]}>
                  <LabelList
                    dataKey="CoT"
                    position="top"
                    formatter={(v) => (v ? `${v}%` : "")}
                    style={{ fontSize: 10, fill: "#6b7280" }}
                  />
                </Bar>
                <Bar dataKey="Agent (TOC)" fill={COLORS["Agent (TOC)"]} radius={[3, 3, 0, 0]}>
                  <LabelList
                    dataKey="Agent (TOC)"
                    position="top"
                    formatter={(v) => (v ? `${v}%` : "")}
                    style={{ fontSize: 10, fill: "#6b7280" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Key finding:</strong> CoT alone does not bridge the gap.
                For Opus, CoT slightly <em>decreases</em> accuracy (−2pp), while tools add +17.5pp.
                The agent lift is primarily from tool access, not reasoning allowance.
              </p>
            </div>
          </div>
        )}

        {activeChart === "ablation" && (
          <div>
            <h2 className="text-lg font-semibold text-gray-800 mb-1">
              TOC vs Raw Output Ablation
            </h2>
            <p className="text-sm text-gray-500 mb-4">
              Agent accuracy with metadata-only TOC vs full raw PHREEQC output
            </p>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={ablationData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                barCategoryGap="25%"
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="model"
                  tick={{ fontSize: 12, fill: "#374151" }}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: 12, fill: "#374151" }}
                  label={{
                    value: "Accuracy (%)",
                    angle: -90,
                    position: "insideLeft",
                    style: { fontSize: 13, fill: "#374151" },
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 13 }} iconType="square" />
                <Bar
                  dataKey="Agent (TOC)"
                  fill="#2563eb"
                  radius={[3, 3, 0, 0]}
                >
                  <LabelList
                    dataKey="Agent (TOC)"
                    position="top"
                    formatter={(v) => `${v}%`}
                    style={{ fontSize: 11, fill: "#1e40af", fontWeight: 600 }}
                  />
                </Bar>
                <Bar
                  dataKey="Agent (Raw)"
                  fill="#f59e0b"
                  radius={[3, 3, 0, 0]}
                >
                  <LabelList
                    dataKey="Agent (Raw)"
                    position="top"
                    formatter={(v) => `${v}%`}
                    style={{ fontSize: 11, fill: "#92400e", fontWeight: 600 }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-4 p-4 bg-amber-50 rounded-lg">
              <p className="text-sm text-amber-800">
                <strong>Finding:</strong> Accuracy is comparable between TOC and raw output (within ±2.5pp).
                TOC reduces per-call execute_phreeqc response size by 82% (3K vs 20K chars)
                and total input tokens by ~22%, enabling higher concurrency.
              </p>
            </div>
          </div>
        )}

        {activeChart === "tokens" && (
          <div>
            <h2 className="text-lg font-semibold text-gray-800 mb-1">
              Token Usage: TOC vs Raw
            </h2>
            <p className="text-sm text-gray-500 mb-4">
              Total input tokens (millions) across 200 questions
            </p>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={tokenData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                barCategoryGap="30%"
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="model"
                  tick={{ fontSize: 12, fill: "#374151" }}
                />
                <YAxis
                  tick={{ fontSize: 12, fill: "#374151" }}
                  label={{
                    value: "Input Tokens (millions)",
                    angle: -90,
                    position: "insideLeft",
                    style: { fontSize: 13, fill: "#374151" },
                  }}
                />
                <Tooltip
                  formatter={(value) => [`${value.toFixed(2)}M tokens`, ""]}
                />
                <Legend wrapperStyle={{ fontSize: 13 }} iconType="square" />
                <Bar dataKey="TOC" fill="#2563eb" radius={[3, 3, 0, 0]}>
                  <LabelList
                    dataKey="TOC"
                    position="top"
                    formatter={(v) => `${v.toFixed(1)}M`}
                    style={{ fontSize: 11, fill: "#1e40af", fontWeight: 600 }}
                  />
                </Bar>
                <Bar dataKey="Raw" fill="#f59e0b" radius={[3, 3, 0, 0]}>
                  <LabelList
                    dataKey="Raw"
                    position="top"
                    formatter={(v) => `${v.toFixed(1)}M`}
                    style={{ fontSize: 11, fill: "#92400e", fontWeight: 600 }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-xs text-blue-600 font-medium uppercase">Per-call reduction</p>
                <p className="text-2xl font-bold text-blue-800">82%</p>
                <p className="text-sm text-blue-600">execute_phreeqc: 3K vs 20K chars</p>
              </div>
              <div className="p-4 bg-amber-50 rounded-lg">
                <p className="text-xs text-amber-600 font-medium uppercase">Total token reduction</p>
                <p className="text-2xl font-bold text-amber-800">~22%</p>
                <p className="text-sm text-amber-600">End-to-end across 200 questions</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}