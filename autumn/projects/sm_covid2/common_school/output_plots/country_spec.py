from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import datetime
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tick

from autumn.core.inputs.database import get_input_db
from autumn.projects.sm_covid2.common_school.calibration import get_bcm_object
from summer.utils import ref_times_to_dti

from autumn.settings.folders import PROJECTS_PATH
from pathlib import Path
import yaml
countries_path = Path(PROJECTS_PATH) / "sm_covid2" / "common_school" / "included_countries.yml"
with countries_path.open() as f:
    INCLUDED_COUNTRIES  = yaml.unsafe_load(f)

REF_DATE = datetime.date(2019,12,31)

def update_rcparams():
    plt.rcParams.update(
        {
            'font.size': 6,
            'axes.titlesize': "large",
            'axes.labelsize': "x-large",
            'xtick.labelsize': 'large',
            'ytick.labelsize': 'large',
            'legend.fontsize': 'large',
            'legend.title_fontsize': 'large',
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

title_lookup = {
    "infection_deaths": "COVID-19 deaths",
    "cumulative_infection_deaths": "Cumulative deaths",
    "cumulative_incidence": "Cumulative incidence",

    "hospital_admissions": "new daily hospital admissions",
    "icu_admissions": "new daily admissions to ICU",
    "incidence": "daily new infections",
    "hospital_admissions": "daily hospital admissions",
    "hospital_occupancy": "Hospital pressure",
    "icu_admissions": "daily ICU admissions",
    "icu_occupancy": "total ICU beds",
    "prop_ever_infected": "ever infected with Delta or Omicron",

    "transformed_random_process": "Transformed random process",

    "peak_hospital_occupancy": "Peak hospital pressure"
}
sc_colours = ["black", "crimson"]
unc_sc_colours = ((0.2, 0.2, 0.8), (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2), (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2))

input_db = get_input_db()
unesco_data = input_db.query(
    table_name='school_closure', 
    columns=["date", "status", "country_id"],
)

SCHOOL_COLORS = {
    'partial': 'azure',
    'full': 'thistle'
}

def y_fmt(tick_val, pos):
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
    else:
        val = tick_val
        if val.is_integer():
            val = int(val)
        return val


def add_school_closure_patches(ax, iso3, ymax, school_colors=SCHOOL_COLORS):
    data = unesco_data[unesco_data['country_id'] == iso3]
    partial_dates = data[data['status'] == "Partially open"]['date'].to_list()
    closed_dates = data[data['status'] == "Closed due to COVID-19"]['date'].to_list()
    
    partial_dates_str = [d.strftime("%Y-%m-%d") for d in partial_dates] 
    closed_dates_str = [d.strftime("%Y-%m-%d") for d in closed_dates] 

    ax.vlines(partial_dates_str,ymin=0, ymax=ymax, lw=1, alpha=1., color=school_colors['partial'], zorder = 1)
    ax.vlines(closed_dates_str, ymin=0, ymax=ymax, lw=1, alpha=1, color=school_colors['full'], zorder = 1)


def plot_model_fit_with_uncertainty(axis, uncertainty_df, output_name, iso3):

    bcm = get_bcm_object(iso3, "main")

    # update_rcparams() 
   
    df = uncertainty_df[(uncertainty_df["scenario"] == "baseline") & (uncertainty_df["type"] == output_name)]

    if output_name in bcm.targets:
        t = bcm.targets[output_name].data
        t.index = ref_times_to_dti(REF_DATE, t.index)
        axis.scatter(list(t.index), t, marker=".", color='black', label='observations', zorder=11, s=3.)
        
    colour = unc_sc_colours[0]      

    median_df = df[df["quantile"] == .5]

    time = median_df['time']
    axis.plot(time, median_df['value'], color=colour, zorder=10, label="model (median)")

    axis.fill_between(
        time, 
        df[df["quantile"] == .25]['value'], df[df["quantile"] == .75]['value'], 
        color=colour, 
        alpha=0.5, 
        edgecolor=None,
        label="model (IQR)"
    )
    axis.fill_between(
        time, 
        df[df["quantile"] == .025]['value'], df[df["quantile"] == .975]['value'], 
        color=colour, 
        alpha=0.3,
        edgecolor=None,
        label="model (95% CI)",
    )

    if output_name == "transformed_random_process":
        axis.set_ylim((0., axis.get_ylim()[1]))

    # axis.tick_params(axis="x", labelrotation=45)
    title = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_ylabel(title)
    plt.tight_layout()

    plt.legend(markerscale=2.)
    axis.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))


def plot_two_scenarios(axis, uncertainty_df, output_name, iso3, include_unc=False):
    # update_rcparams()

    ymax = 0.
    for i_sc, scenario in enumerate(["baseline", "scenario_1"]):
        df = uncertainty_df[(uncertainty_df["scenario"] == scenario) & (uncertainty_df["type"] == output_name)]
        median_df = df[df["quantile"] == .5]
        time = median_df['time']
        
        colour = unc_sc_colours[i_sc]
        label = "baseline" if i_sc == 0 else "schools open"
        scenario_zorder = 10 if i_sc == 0 else i_sc + 2

        if include_unc:
            axis.fill_between(
                time, 
                df[df["quantile"] == .25]['value'], df[df["quantile"] == .75]['value'], 
                color=colour, alpha=0.7, 
                # label=interval_label,
                zorder=scenario_zorder
            )
            ymax = max(ymax, df[df["quantile"] == .75]['value'].max())
        else:
            ymax = median_df['value'].max()

        axis.plot(time, median_df['value'], color=colour, label=label, lw=1.)
        
    plot_ymax = ymax * 1.1    
    add_school_closure_patches(axis, iso3, ymax=plot_ymax)

    # axis.tick_params(axis="x", labelrotation=45)
    title = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_ylabel(title)
    # axis.set_xlim((model_start, model_end))
    axis.set_ylim((0, plot_ymax))

    axis.legend()

    axis.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

    plt.tight_layout()


def plot_final_size_compare(axis, uncertainty_df, output_name):
    # update_rcparams()
    # plt.rcParams.update({'font.size': 12})    
    box_width = .5
    color = 'black'
    box_color= 'lightcoral'
    y_max = 0
    for i, scenario in enumerate(["baseline", "scenario_1"]):
      
        df = uncertainty_df[(uncertainty_df["scenario"] == scenario) & (uncertainty_df["type"] == output_name) & (uncertainty_df["time"] == uncertainty_df["time"].max())]

        x = 1 + i
        # median
        axis.hlines(y=df[df["quantile"] == .5]['value'], xmin=x - box_width / 2. , xmax= x + box_width / 2., lw=1., color=color, zorder=3)    
        
        # IQR
        q_75 = float(df[df["quantile"] == .75]['value'])
        q_25 = float(df[df["quantile"] == .25]['value'])
        rect = Rectangle(xy=(x - box_width / 2., q_25), width=box_width, height=q_75 - q_25, zorder=2, facecolor=box_color)
        axis.add_patch(rect)

        # 95% CI
        q_025 = float(df[df["quantile"] == .025]['value'])
        q_975 = float(df[df["quantile"] == .975]['value'])
        axis.vlines(x=x, ymin=q_025 , ymax=q_975, lw=.7, color=color, zorder=1)

        y_max = max(y_max, q_975)
        
    title = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_ylabel(title)
    axis.set_xticks(ticks=[1, 2], labels=["baseline", "schools open"]) #, fontsize=15)

    axis.set_xlim((0.5, 2.5))
    axis.set_ylim((0, y_max * 1.2))

    axis.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))


def plot_diff_outputs(axis, diff_quantiles_df, output_names):

    xlab_lookup = {
        "cases_averted_relative": "infections", 
        "deaths_averted_relative": "deaths",
        "delta_hospital_peak_relative": "hospital pressure"
    }

    box_width = .2
    med_color = 'white'
    box_color= 'black'
    y_max_abs = 0.
    for i, diff_output in enumerate(output_names): 

        data = - 100. * diff_quantiles_df[diff_output] # use %. And use "-" so positive nbs indicate positive effect of closures
        x = 1 + i
        # median
        axis.hlines(y=data.loc[0.5], xmin=x - box_width / 2. , xmax= x + box_width / 2., lw=2., color=med_color, zorder=3)    
        
        # IQR
        q_75 = data.loc[0.75]
        q_25 = data.loc[0.25]
        rect = Rectangle(xy=(x - box_width / 2., q_25), width=box_width, height=q_75 - q_25, zorder=2, facecolor=box_color)
        axis.add_patch(rect)

        # 95% CI
        q_025 = data.loc[0.025]
        q_975 = data.loc[0.975]
        axis.vlines(x=x, ymin=q_025 , ymax=q_975, lw=1.5, color=box_color, zorder=1)

        y_max_abs = max(abs(q_975), y_max_abs)
        y_max_abs = max(abs(q_025), y_max_abs)
 
    # title = output_name if output_name not in title_lookup else title_lookup[output_name]
    
    y_label = "% peak reduction" if "delta_hospital_peak_relative" in output_names else "% averted by school closure"
    axis.set_ylabel(y_label)
    

    labels = [xlab_lookup[o] for o in output_names]
    axis.set_xticks(ticks=range(1, len(output_names) + 1), labels=labels) #, fontsize=15)

    axis.set_xlim((0.5, len(output_names) + 1.5))
    axis.set_ylim(-1.2*y_max_abs, 1.2*y_max_abs)
    
    # add coloured backgorund patches
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim() 
    rect_up = Rectangle(xy=(xmin, 0.), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="honeydew")
    axis.add_patch(rect_up)
    rect_low = Rectangle(xy=(xmin, ymin), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="mistyrose")
    axis.add_patch(rect_low)

    axis.text(len(output_names) + .3, ymax / 2., s="Positive effect of\nschool closures")
    axis.text(len(output_names) + .3, ymin / 2., s="Negative effect of\nschool closures")

def format_date_axis(axis):
    axis.set_xticks(
        [datetime.datetime(2020,1,1), datetime.datetime(2021,1,1), datetime.datetime(2022,1,1), datetime.datetime(2023,1,1)],
        ["Jan 2020", "Jan 2021", "Jan 2022", "Jan 2023"]
    )

def remove_axes_box(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def make_country_output_tiling(iso3, uncertainty_df, diff_quantiles_df, output_folder):
    country_name = INCLUDED_COUNTRIES['all'][iso3]

    update_rcparams()
    plt.style.use("default")
    fig = plt.figure(figsize=(8.3, 11.7), dpi=300) # crete an A4 figure
    outer = gridspec.GridSpec(
        3, 1, hspace=.15, height_ratios=(3, 71, 26), 
        left=0.125, right=0.97, bottom=0.06, top =.97   # this affects the outer margins of the saved figure 
    )

    #### Top row with country name
    ax1 = fig.add_subplot(outer[0, 0])
    t = ax1.text(0.5,0.5, country_name, fontsize=16)
    t.set_ha('center')
    t.set_va('center')
    ax1.set_xticks([])
    ax1.set_yticks([])

    #### Second row will need to be split
    outer_cell = outer[1, 0]
    # first split in left/right panels
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_cell, wspace=.3, width_ratios=(70, 30))
    left_grid = inner_grid[0, 0]  # will contain timeseries plots
    right_grid = inner_grid[0, 1]  # will contain final size plots

    #### Split left panel into 3 panels
    inner_left_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=left_grid, hspace=.15, height_ratios=(1, 1, 1))
    # calibration
    ax2 = fig.add_subplot(inner_left_grid[0, 0])
    plot_model_fit_with_uncertainty(ax2, uncertainty_df, "infection_deaths", iso3)
    format_date_axis(ax2)
    remove_axes_box(ax2)
    # plt.setp(ax2.get_xticklabels(), visible=False)
    # scenario compare deaths
    ax3 = fig.add_subplot(inner_left_grid[1, 0]) #, sharex=ax2)
    plot_two_scenarios(ax3, uncertainty_df, "infection_deaths", iso3, True)
    format_date_axis(ax3)
    remove_axes_box(ax3)

    # plt.setp(ax3.get_xticklabels(), visible=False)
    # scenario compare hosp
    ax4 = fig.add_subplot(inner_left_grid[2, 0])  #, sharex=ax2)
    plot_two_scenarios(ax4, uncertainty_df, "hospital_occupancy", iso3, True)
    format_date_axis(ax4)
    remove_axes_box(ax4)

    ## Split right panel into 3 panels
    inner_right_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=right_grid, hspace=.15, height_ratios=(1, 1, 1))
    # final size incidence
    ax5 = fig.add_subplot(inner_right_grid[0, 0])
    plot_final_size_compare(ax5, uncertainty_df, "cumulative_incidence")
    remove_axes_box(ax5)

    # final size deaths
    ax6 = fig.add_subplot(inner_right_grid[1, 0]) #, sharex=ax5)
    plot_final_size_compare(ax6, uncertainty_df, "cumulative_infection_deaths")
    remove_axes_box(ax6)

    # # hosp peak
    ax7 = fig.add_subplot(inner_right_grid[2, 0])  #, sharex=ax5)
    plot_final_size_compare(ax7, uncertainty_df, "peak_hospital_occupancy")
    # ax7.set_xticks(ticks=[1, 2], labels=["baseline", "schools\nopen"]) #, fontsize=15)
    remove_axes_box(ax7)

    #### Third row will need to be split into 6 panels
    outer_cell = outer[2, 0]
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_cell, wspace=.25, hspace=.05, width_ratios=(56, 44), height_ratios=(15, 85))

    # top left
    ax_tl = fig.add_subplot(inner_grid[0, 0])
    # t = ax_tl.text(0.5,0.5, "Age-specific incidence (baseline scenario)", fontsize=12)
    t = ax_tl.text(0.5,0.5, "Relative impact of school closure", fontsize=11)
    t.set_ha('center')
    t.set_va('center')
    ax_tl.set_xticks([])
    ax_tl.set_yticks([])

    # top right
    ax_tr = fig.add_subplot(inner_grid[0, 1])
    # t = ax_tr.text(0.5,0.5, "Age-specific incidence (schools open)", fontsize=12)
    t = ax_tr.text(0.5,0.5, "Relative impact of school closure", fontsize=11)
    t.set_ha('center')
    t.set_va('center')
    ax_tr.set_xticks([])
    ax_tr.set_yticks([])

    # middle left
    ax8 = fig.add_subplot(inner_grid[1, 0])
    plot_diff_outputs(ax8, diff_quantiles_df, ["cases_averted_relative", "deaths_averted_relative"])
    remove_axes_box(ax8)
    # plot_incidence_by_age(derived_outputs, ax8, 0, as_proportion=False)
    # plt.setp(ax8.get_xticklabels(), visible=False)

    # middle right
    ax9 = fig.add_subplot(inner_grid[1, 1])
    plot_diff_outputs(ax9, diff_quantiles_df, ["delta_hospital_peak_relative"])
    remove_axes_box(ax9)

    # plot_incidence_by_age(derived_outputs, ax9, 1, as_proportion=False)
    # plt.setp(ax9.get_xticklabels(), visible=False)

    # # bottom left
    # ax10 = fig.add_subplot(inner_grid[2, 0], sharex=ax8)
    # plot_incidence_by_age(derived_outputs, ax10, 0, as_proportion=True)
    # # bottom right
    # ax11 = fig.add_subplot(inner_grid[2, 1], sharex=ax9)
    # plot_incidence_by_age(derived_outputs, ax11, 1, as_proportion=True)

    fig.savefig(os.path.join(output_folder, "tiling.png"), facecolor="white")
    fig.savefig(os.path.join(output_folder, "tiling.pdf"), facecolor="white")

    plt.close()
