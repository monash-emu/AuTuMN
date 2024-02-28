import pandas as pd
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt 
from matplotlib.patches import Rectangle

from autumn.projects.sm_covid2.common_school.runner_tools import INCLUDED_COUNTRIES
from autumn.models.sm_covid2.stratifications.strains import get_first_variant_report_date
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tick
from copy import copy

from autumn.projects.sm_covid2.common_school.output_plots.country_spec import (
    format_date_axis, 
    remove_axes_box, 
    plot_model_fit_with_uncertainty, 
    plot_incidence_by_age, 
    add_school_closure_patches, 
    y_fmt, 
    plot_final_size_compare, 
    unesco_data,
    SCHOOL_COLORS,
    unc_sc_colours,
    title_lookup,
)


def ad_panel_number(ax, panel_number, x=-0.1, y=1.1):
    ax.text(x, y, panel_number, transform=ax.transAxes, fontsize=10, va='top', ha='right')

def _add_school_closure_patches(ax, iso3, scenario, school_colors=SCHOOL_COLORS, ymin_asprop=0., ymax_asprop=1., txt=""):
    data = unesco_data[unesco_data['country_id'] == iso3]
    partial_dates = data[data['status'] == "Partially open"]['date'].to_list()
    closed_dates = data[data['status'] == "Closed due to COVID-19"]['date'].to_list()
    academic_dates = data[data['status'] == "Academic break"]['date'].to_list()
    
    partial_dates_str = [d.strftime("%Y-%m-%d") for d in partial_dates] 
    closed_dates_str = [d.strftime("%Y-%m-%d") for d in closed_dates] 
    academic_dates_str = [d.strftime("%Y-%m-%d") for d in academic_dates] 

    plot_ymin, plot_ymax = ax.get_ylim()
    ymin = plot_ymin + ymin_asprop * (plot_ymax - plot_ymin)
    ymax = plot_ymin + ymax_asprop * (plot_ymax - plot_ymin)

    if scenario == 'baseline':
        ax.vlines(partial_dates_str,ymin=ymin, ymax=ymax, lw=1, alpha=1., color=school_colors['partial'], zorder = 1)
        ax.vlines(closed_dates_str, ymin=ymin, ymax=ymax, lw=1, alpha=1, color=school_colors['full'], zorder = 1)
    ax.vlines(academic_dates_str,ymin=ymin, ymax=ymax, lw=1, alpha=1., color=school_colors['academic'], zorder = 1)

    plot_xmin, plot_xmax = ax.get_xlim()
    ax.text(x=plot_xmin + 0.015 * (plot_xmax - plot_xmin), y=ymin - (plot_ymax - plot_ymin) * .02, s=txt, size=6, va="bottom", ha="left")

def add_variant_emergence(ax, iso3):
    linestyles = {"delta": "dashed", "omicron": "dotted"}
    plot_ymin, plot_ymax = ax.get_ylim()

    for voc_name in ["delta", "omicron"]:
        d = get_first_variant_report_date(voc_name, iso3)
        ax.vlines(x=d, ymin=plot_ymin, ymax=plot_ymax, linestyle=linestyles[voc_name], color="grey")

def add_vacc_coverage(ax, uncertainty_dfs):
    vacc_axis = ax.twinx()
    vacc_axis.plot(100.* uncertainty_dfs['baseline']['prop_immune_vaccinated']['0.5'], ls='-.', lw=1, color='black', zorder=-100, label="vaccine coverage")
    
    vacc_axis.set_ylim((0., 100.))
    vacc_axis.set_ylabel("% vaccinated")
    vacc_axis.yaxis.set_label_coords(1.05, .52)
    vacc_axis.spines['top'].set_visible(False)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = vacc_axis.get_legend_handles_labels()
    vacc_axis.legend(lines + lines2, labels + labels2, loc='lower right', labelspacing=.3)


AGE_COLOURS = ["cornflowerblue", "slateblue", "mediumseagreen", "lightcoral", "purple"]

def _plot_incidence_by_age(derived_outputs, ax, scenario, as_proportion: bool, legend=False):
   
    y_label = "Incidence prop." if as_proportion else "Daily infections"    

    scenario_name = {"baseline": "historical", "scenario_1": "counterfactual"}
    y_label += f"\n({scenario_name[scenario]})"

    times = derived_outputs[scenario]["incidence"].index.to_list()
    running_total = [0] * len(derived_outputs[scenario]["incidence"])
    age_groups = [0, 15, 25, 50, 70]

    y_max = 1. if as_proportion else max([derived_outputs[sc]["incidence"].max() for sc in ["baseline", "scenario_1"]])

    for i_age, age_group in enumerate(age_groups):
        output_name = f"incidenceXagegroup_{age_group}"
    
        if i_age < len(age_groups) - 1:
            upper_age = age_groups[i_age + 1] - 1 if i_age < len(age_groups) - 1 else ""
            age_group_name = f"{age_group}-{upper_age}"
        else:
            age_group_name = f"{age_group}+"

        age_group_incidence = derived_outputs[scenario][output_name]
        
        if as_proportion:
            numerator, denominator = age_group_incidence, derived_outputs[scenario]["incidence"]
            age_group_proportion = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
            new_running_total = age_group_proportion + running_total
        else: 
            new_running_total = age_group_incidence + running_total 

        ax.fill_between(times, running_total, new_running_total, color=AGE_COLOURS[i_age], label=age_group_name, zorder=2, alpha=.8)
        running_total = copy(new_running_total)

    y_max = max(new_running_total)
    plot_ymax = y_max * 1.1

    # work out first time with positive incidence
    t_min, t_max = derived_outputs["baseline"]['incidence'].gt(0).idxmax(), derived_outputs["baseline"].index.max()    

    ax.set_xlim((t_min, t_max))
    # ax.set_ylim((0, plot_ymax))

    ax.set_ylabel(y_label)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            reversed(handles),
            reversed(labels),
            # title="Age:",
            # fontsize=12,
            # title_fontsize=12,
            labelspacing=.2,
            handlelength=1.,
            handletextpad=.5,
            columnspacing=1.,
            facecolor="white",
            ncol=1,
            bbox_to_anchor=(1.05, 1.05)
        )


def plot_inc_by_strain(derived_outputs, ax, as_prop=False, legend=False):
    
    y_label = "Infection proportion" if as_prop else "N infections by strain"

    output_name = "cumulative_incidence_prop" if as_prop else "cumulative_incidence"     
    strain_data = {s: [derived_outputs[sc][f"{output_name}Xstrain_{s}"].iloc[-1] for sc in ["baseline", "scenario_1"]] for s in ["wild_type", "delta", "omicron"]} 
    df = pd.DataFrame(strain_data, index = ["Historical", "Counterfactual"])
    df = df.rename(columns={"wild_type": "wild type"})

    df.plot.bar(
        stacked=True, 
        ax=ax, 
        # color={"wild type": "lightsteelblue", "delta": "lightcoral", "omicron": "mediumseagreen"}, 
        color={"wild type": "lightgrey", "delta": "lightcoral", "omicron": "dodgerblue"}, 
        rot=0
    )    

    ax.set_ylabel(y_label)

    if legend:
        ax.legend(
            labelspacing=.2,
            handlelength=1.,
            handletextpad=.5,
            columnspacing=1.,
            facecolor="white",
            ncol=2,
            # bbox_to_anchor=(1.05, 1.05)
        )
    else:
        ax.get_legend().remove()



def cum_over50_fmt(tick_val):
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        if val.is_integer():
            val = int(val)
        return f"{val}G"
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        if val.is_integer():
            val = int(val)
        return f"{val}M"
    elif tick_val >= 1000:
        val = round(tick_val / 1000, 1)
        if val.is_integer():
            val = int(val)
        return f"{val}K"
    elif 0. < tick_val < 1.:
        return round(tick_val, 2)
    else:
        val = tick_val
        if val.is_integer():
            val = int(val)
        return val


def plot_cum_incidence_by_age(derived_outputs, ax, legend=False):
    y_label = "N infections by age"
    age_groups = [0, 15, 25, 50, 70]

    age_inc_data = {str(agegroup): [derived_outputs[sc][f"incidenceXagegroup_{agegroup}"].sum() for sc in ["baseline", "scenario_1"]] for agegroup in age_groups} 
    
    df = pd.DataFrame(age_inc_data, index = ["Historical", "Counterfactual"])

    column_names, colors = {}, {}
    for i_age, age_group in enumerate(age_groups):
        
        if i_age < len(age_groups) - 1:
            upper_age = age_groups[i_age + 1] - 1 if i_age < len(age_groups) - 1 else ""
            c_name = f"{age_group}-{upper_age}"
        else:
            c_name = f"{age_group}+"

        column_names[str(age_group)] = c_name
        colors[c_name] = AGE_COLOURS[i_age]

    df = df.rename(columns=column_names)

    df.plot.bar(
        stacked=True, 
        ax=ax, 
        color=colors, 
        alpha=.8,
        rot=0
    )    

    ax.set_ylabel(y_label)

    for i_sc, sc in enumerate(["baseline", "scenario_1"]):
        total_inc = derived_outputs[sc]["incidence"].sum()
        n_under_50 = (derived_outputs[sc][f"incidenceXagegroup_0"] + derived_outputs[sc][f"incidenceXagegroup_15"] +  derived_outputs[sc][f"incidenceXagegroup_25"]).sum()
        n_over_50 = total_inc - n_under_50

        ax.plot([i_sc + .28, i_sc + .28], [n_under_50, total_inc], marker='_', lw=0, color='black', ms=3.)
        ax.text(x=i_sc + .34, y=0.5 * (n_under_50 + total_inc), s=cum_over50_fmt(n_over_50), fontsize=7, rotation=90, va='center')

    if legend:
        ax.legend(
            labelspacing=.2,
            handlelength=1.,
            handletextpad=.5,
            columnspacing=1.,
            facecolor="white",
            ncol=2,
            # bbox_to_anchor=(1.05, 1.05)
        )
    else:
        ax.get_legend().remove()


def _plot_two_scenarios(axis, uncertainty_dfs, output_name, iso3, include_unc=False, include_legend=True):

    ymax = 0.
    for i_sc, scenario in enumerate(["baseline", "scenario_1"]):
        df = uncertainty_dfs[scenario][output_name]
        median_df = df['0.5']
        time = df.index
        
        colour = unc_sc_colours[i_sc]
        label = "historical" if i_sc == 0 else "counterfactual"
        scenario_zorder = 10 if i_sc == 0 else i_sc + 2

        if include_unc:
            axis.fill_between(
                time, 
                df['0.25'], df['0.75'], 
                color=colour, alpha=0.7, 
                edgecolor=None,
                # label=interval_label,
                zorder=scenario_zorder
            )
            ymax = max(ymax, df['0.75'].max())
        else:
            ymax = median_df.max()

        axis.plot(time, median_df, color=colour, label=label, lw=1.)
        
    plot_ymax = ymax * 1.2    

    _add_school_closure_patches(axis, iso3, "baseline", ymin_asprop=1.095, ymax_asprop=1.2, txt="historic.")
    _add_school_closure_patches(axis, iso3, "scenario_1", ymin_asprop=.8, ymax_asprop=.85, txt= "counterfac.")

    # axis.tick_params(axis="x", labelrotation=45)
    title = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_ylabel(title)
    # axis.set_xlim((model_start, model_end))
    axis.set_ylim((0, plot_ymax))

    if include_legend:
        location = "lower right" if output_name == "prop_ever_infected" else "upper right"
        # axis.legend(title="(median and IQR)")
        axis.legend(
            handletextpad=.5,
            handlelength=1,
            loc=location
        )

    axis.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

    # plt.tight_layout()


def _plot_diff_outputs(axis, diff_quantiles_df, output_names):

    xlab_lookup = {
        "cases_averted_relative": "Infections", 
        "deaths_averted_relative": "Deaths",
        "delta_hospital_peak_relative": "Hospital\npressure"
    }

    box_width = .2
    med_color = 'white'
    box_color= 'black'
    y_max_abs = 0.
    for i, diff_output in enumerate(output_names): 

        assert diff_output.endswith("_relative"), "function only compatible with relative outputs"

        data = - 100. * diff_quantiles_df[diff_output] # use %. And use "-" so positive nbs indicate positive effect of closures
        x = 1 + i
        # median
        axis.hlines(y=data.loc[0.5], xmin=x - box_width / 2. , xmax= x + box_width / 2., lw=.8, color=med_color, zorder=3)    
        
        # IQR
        q_75 = data.loc[0.75]
        q_25 = data.loc[0.25]
        rect = Rectangle(xy=(x - box_width / 2., q_25), width=box_width, height=q_75 - q_25, zorder=2, facecolor=box_color)
        axis.add_patch(rect)

        # 95% CI
        q_025 = data.loc[0.025]
        q_975 = data.loc[0.975]
        axis.vlines(x=x, ymin=q_025 , ymax=q_975, lw=1, color=box_color, zorder=1)

        y_max_abs = max(abs(q_975), y_max_abs)
        y_max_abs = max(abs(q_025), y_max_abs)
 
    # title = output_name if output_name not in title_lookup else title_lookup[output_name]
    
    y_label = "% Outcome reduction"
    axis.set_ylabel(y_label)
    
    labels = [xlab_lookup[o] for o in output_names]
    axis.set_xticks(ticks=range(1, len(output_names) + 1), labels=labels) #, fontsize=15)

    axis.set_xlim((0.5, len(output_names) + 1))
    axis.set_ylim(-1.2*y_max_abs, 1.2*y_max_abs)
    
    # add coloured backgorund patches
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim() 
    rect_up = Rectangle(xy=(xmin, 0.), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="white")
    axis.add_patch(rect_up)
    rect_low = Rectangle(xy=(xmin, ymin), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="gainsboro")
    axis.add_patch(rect_low)

    axis.text(len(output_names) + .25, ymax / 2., s="positive\neffect")
    axis.text(len(output_names) + .25, ymin / 2., s="negative\neffect")



def make_country_highlight_figure(iso3, uncertainty_dfs, diff_quantiles_df, derived_outputs, include_country_name=False):

    plt.rcParams.update(
        {
            'font.family':"Times New Roman",  
            'font.size': 6,
            'axes.titlesize': "large",
            'axes.labelsize': "large",
            'xtick.labelsize': 'large',
            'ytick.labelsize': 'large',
            'legend.fontsize': 7,  # 'large', # 'medium',
            'legend.title_fontsize': 7, # 'large',
            'lines.linewidth': 1.,

            'xtick.major.size':    2.5,
            'xtick.major.width':   0.6,
            'xtick.major.pad':     2,

            'ytick.major.size':    2.5,
            'ytick.major.width':   0.6,
            'ytick.major.pad':     2,

            'axes.labelpad':      2.
        }
    )

    country_name = INCLUDED_COUNTRIES['all'][iso3]
    fig = plt.figure(figsize=(10.5, 5), dpi=300) # crete an A4 figure
    if include_country_name:
        n_outer_rows = 2
        height_ratios = [3, 97]
    else:
        n_outer_rows = 1
        height_ratios = [100]

    super_outer = gridspec.GridSpec(
        n_outer_rows, 1, height_ratios=height_ratios,
        left=0.125, right=0.97, bottom=0.06, top =.97   # this affects the outer margins of the saved figure 
    )
    #### Top row with country name
    if include_country_name:
        ax1 = fig.add_subplot(super_outer[0, 0])
        t = ax1.text(0.5,0.5, country_name, fontsize=16)
        t.set_ha('center')
        t.set_va('center')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)

    outer = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=super_outer[n_outer_rows - 1, 0], wspace=.29, width_ratios=(41, 41, 18)
    )    

    # LEFT column
    outer_cell = outer[0, 0]
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_cell, hspace=.3, height_ratios=(30, 45, 100. - 30 - 45))

    # Top Left: deaths fit
    death_fit_ax = fig.add_subplot(inner_grid[0, 0])
    plot_model_fit_with_uncertainty(death_fit_ax, uncertainty_dfs['baseline'], "infection_deaths_ma7", iso3)
    add_variant_emergence(death_fit_ax, iso3)    
    format_date_axis(death_fit_ax)
    remove_axes_box(death_fit_ax)
    ad_panel_number(death_fit_ax, "A")

    # if iso3 in INCLUDED_COUNTRIES['national_sero']:
    #     insert_inside_plot_for_sero(death_fit_ax, uncertainty_dfs['baseline'], iso3)


    # Middle Left: incidence by age
    inner_inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=inner_grid[1, 0], hspace=.05)
    age_inc_baseline_ax = fig.add_subplot(inner_inner_grid[0, 0])
    _plot_incidence_by_age(derived_outputs, age_inc_baseline_ax, "baseline", as_proportion=True, legend=True)    
    _add_school_closure_patches(age_inc_baseline_ax, iso3, "baseline", ymin_asprop=1., ymax_asprop=1.2)
    add_variant_emergence(age_inc_baseline_ax, iso3)
    age_inc_baseline_ax.get_xaxis().set_visible(False)
    ad_panel_number(age_inc_baseline_ax, "B", y=1.15)

    age_inc_sc1_ax = fig.add_subplot(inner_inner_grid[1, 0]) #, sharex=age_inc_baseline_ax)
    _plot_incidence_by_age(derived_outputs, age_inc_sc1_ax, "scenario_1", as_proportion=True)   
    _add_school_closure_patches(age_inc_sc1_ax, iso3, "scenario_1", ymin_asprop=1., ymax_asprop=1.2)
    add_variant_emergence(age_inc_sc1_ax, iso3)

    for ax in [age_inc_baseline_ax, age_inc_sc1_ax]:
        format_date_axis(ax)
        remove_axes_box(ax)

    # Bottom Left: Inc prop by strain
    inner_inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=inner_grid[2, 0], wspace=.5)

    inc_prop_age_ax = fig.add_subplot(inner_inner_grid[0, 0])
    plot_cum_incidence_by_age(derived_outputs, inc_prop_age_ax)
    inc_prop_age_ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    ad_panel_number(inc_prop_age_ax, "C", x=-0.25)

    inc_prop_strain_ax = fig.add_subplot(inner_inner_grid[0, 1])
    plot_inc_by_strain(derived_outputs, inc_prop_strain_ax, False, legend=True)
    inc_prop_strain_ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

    # MIDDLE Column
    outer_cell = outer[0, 1]
    inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_cell, hspace=.3)

    # Now all the right panel plots for scenario comparisons
    for i_output, output in enumerate(["incidence", "hospital_occupancy", "infection_deaths_ma7", "prop_ever_infected"]):
        include_main_ax_legend = i_output != 3
        sc_compare_ax = fig.add_subplot(inner_grid[i_output, 0])
        _plot_two_scenarios(sc_compare_ax, uncertainty_dfs, output, iso3, include_unc=True, include_legend=include_main_ax_legend)
        add_variant_emergence(sc_compare_ax, iso3)

        if i_output == 3:
            add_vacc_coverage(sc_compare_ax, uncertainty_dfs)

        format_date_axis(sc_compare_ax)
        remove_axes_box(sc_compare_ax)
        ad_panel_number(sc_compare_ax, ["D", "E", "F", "G"][i_output])




    # RIGHT Column
    outer_cell = outer[0, 2]
    inner_grid = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_cell, hspace=.3)

    # Now all the right panel plots for scenario comparisons
    for i_output, output in enumerate(["cumulative_incidence", "peak_hospital_occupancy", "cumulative_infection_deaths"]): 
        boxplot_ax = fig.add_subplot(inner_grid[i_output, 0])
        plot_final_size_compare(boxplot_ax, uncertainty_dfs, output)
        remove_axes_box(boxplot_ax) 
        ad_panel_number(boxplot_ax, ["H", "I", "J"][i_output], x=-.33)
        
    diff_outputs_ax = fig.add_subplot(inner_grid[3, 0])
    _plot_diff_outputs(diff_outputs_ax, diff_quantiles_df, ["cases_averted_relative", "deaths_averted_relative", "delta_hospital_peak_relative"])
    remove_axes_box(diff_outputs_ax)
    ad_panel_number(diff_outputs_ax, "K", x=-.27)

    return fig