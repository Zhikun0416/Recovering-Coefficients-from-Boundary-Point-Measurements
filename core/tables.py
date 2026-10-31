"""Write the six numerical tables in manuscript order."""

from .experiments import NOISE_LEVELS, STRATEGIES


def percent(value):
    return f"{100 * value:.2f}"


def _table(title, headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in rows))
              for i, header in enumerate(headers)]

    def line(values):
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    return "\n".join([
        title, line(headers), "-+-".join("-" * width for width in widths),
        *(line(row) for row in rows),
    ])


def write_tables(summaries, path):
    """Preserve manuscript rounding and the order of all six tables."""
    tables = []
    rows = []
    for index, name in enumerate(("q0", "q2", "a0", "a2")):
        first, second = [summaries[(case, 1)]["cross_finer"] for case in (1, 2)]
        rows.append([
            name, f"{first['truth'][index]:g}",
            f"{first['recovered'][index]:.4f}", percent(first["relative_error"][index]),
            f"{second['recovered'][index]:.4f}", percent(second["relative_error"][index]),
        ])
    tables.append(_table(
        "Table 1. EX1: refined-grid coefficient recovery.",
        ["Parameter", "Truth", "Case1 recovered", "Case1 error (%)",
         "Case2 recovered", "Case2 error (%)"], rows,
    ))
    rows = []
    for case in (1, 2):
        for stage, label in (("cross_raw", "Base"), ("cross_finer", "Refined")):
            fit = summaries[(case, 2)][stage]
            rows.append([f"Case{case}", label, *[
                percent(fit[key]) for key in (
                    "maximum_parameter_relative_error", "q_relative_L2", "coefficient_relative_L2",
                )
            ]])
    tables.append(_table(
        "Table 2. EX2: inverse-grid refinement with fixed fine-grid data.",
        ["Case", "Inverse grid", "E_inf (%)", "E_2(q) (%)", "E_2(a) (%)"], rows,
    ))
    rows = []
    for case in (1, 2):
        for key, label in (("quadratic_control_comparison", "Quadratic"),
                           ("clean_cubic_comparison", "Cubic")):
            comparison = summaries[(case, 3)][key]
            values = []
            for model in ("linear", "quadratic", "cubic"):
                value = comparison["candidates"][model]["validation_rms"]
                values.append(f"{value:.4e}" + ("*" if model == comparison["selected_model"] else ""))
            rows.append([f"Case{case}", label, *values])
    tables.append(_table(
        "Table 3. EX3: independent validation RMS (* marks the selected model).",
        ["Case", "True model", "Linear", "Quadratic", "Cubic"], rows,
    ))
    rows = []
    for case in (1, 2):
        for trial in summaries[(case, 3)]["order_selection"]:
            rows.append([f"Case{case}", f"{trial['noise_level']:.0e}", *[
                trial["counts"][model] for model in ("linear", "quadratic", "cubic")
            ]])
    tables.append(_table(
        "Table 4. EX3: model-selection counts over ten noisy trials.",
        ["Case", "Noise level", "Linear", "Quadratic", "Cubic"], rows,
    ))
    rows = []
    labels = {"f_only": "f only", "g_only": "g only", "dual": "Both f,g"}
    for case in (1, 2):
        for strategy in STRATEGIES:
            group = summaries[(case, 4)]["strategies"][strategy]
            fit, sensitivity = group["cross_finer"], group["sensitivity"]
            rows.append([
                f"Case{case}", labels[strategy],
                f"{sensitivity['smallest_singular_value']:.4e}",
                f"{sensitivity['condition_number']:.4f}",
                percent(fit["parameter_relative_RMS"]),
                percent(fit["maximum_parameter_relative_error"]),
            ])
    tables.append(_table(
        "Table 5. EX4: base-grid sensitivities and refined-grid recovery.",
        ["Case", "Inputs", "sigma_min(J)", "cond(J)", "E_RMS (%)", "E_inf (%)"], rows,
    ))
    rows = []
    for case in (1, 2):
        noise = summaries[(case, 4)]["noise"]
        for level in NOISE_LEVELS:
            values = [next(row["mean_parameter_relative_RMS"] for row in noise
                           if row["noise_level"] == level and row["strategy"] == strategy)
                      for strategy in STRATEGIES]
            rows.append([f"Case{case}", f"{level:.0e}", *[percent(value) for value in values]])
    tables.append(_table(
        "Table 6. EX4: mean parameter relative RMS error over ten noisy trials.",
        ["Case", "Noise level", "f only (%)", "g only (%)", "Both f,g (%)"], rows,
    ))
    path.write_text("\n\n".join(tables) + "\n", encoding="ascii")
