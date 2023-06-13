from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import datetime

from autumn.core.inputs.database import get_input_db
from autumn.projects.sm_covid2.common_school.calibration import get_bcm_object
from summer.utils import ref_times_to_dti

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
    "cumulative_infection_deaths": "Cumulative COVID-19 deaths",
    "cumulative_incidence": "Cumulative COVID-19 incidence",

    "hospital_admissions": "new daily hospital admissions",
    "icu_admissions": "new daily admissions to ICU",
    "incidence": "daily new infections",
    "hospital_admissions": "daily hospital admissions",
    "hospital_occupancy": "total hospital beds",
    "icu_admissions": "daily ICU admissions",
    "icu_occupancy": "total ICU beds",
    "prop_ever_infected": "ever infected with Delta or Omicron",

    "transformed_random_process": "Transformed random process",

    "peak_hospital_occupancy": "Peak COVID-19 hospital occupancy"
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

def add_school_closure_patches(ax, iso3, ymax, school_colors=SCHOOL_COLORS):
    data = unesco_data[unesco_data['country_id'] == iso3]
    partial_dates = data[data['status'] == "Partially open"]['date'].to_list()
    closed_dates = data[data['status'] == "Closed due to COVID-19"]['date'].to_list()
    
    partial_dates_str = [d.strftime("%Y-%m-%d") for d in partial_dates] 
    closed_dates_str = [d.strftime("%Y-%m-%d") for d in closed_dates] 

    ax.vlines(partial_dates_str,ymin=0, ymax=ymax, lw=1, alpha=1., color=school_colors['partial'], zorder = 1)
    ax.vlines(closed_dates_str, ymin=0, ymax=ymax, lw=1, alpha=1, color=school_colors['full'], zorder = 1)


def plot_model_fit(axis, uncertainty_df, output_name, iso3):

    bcm = get_bcm_object(iso3, "main")

    update_rcparams() 
   
    df = uncertainty_df[(uncertainty_df["scenario"] == "baseline") & (uncertainty_df["type"] == output_name)]

    if output_name in bcm.targets:
        t = bcm.targets[output_name].data
        t.index = ref_times_to_dti(REF_DATE, t.index)
        axis.scatter(list(t.index), t, marker=".", color='black', label='observations', zorder=11, s=3.)
        
    colour = unc_sc_colours[0]      

    median_df = df[df["quantile"] == .5]

    time = [datetime.datetime.strptime(s, "%Y-%m-%d") for s in median_df['time']]
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

    axis.tick_params(axis="x", labelrotation=45)
    title = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_ylabel(title)
    plt.tight_layout()

    plt.legend(markerscale=2.)


def plot_two_scenarios(axis, uncertainty_df, output_name, iso3, include_unc=False):
    update_rcparams()

    ymax = 0.
    for i_sc, scenario in enumerate(["baseline", "scenario_1"]):
        df = uncertainty_df[(uncertainty_df["scenario"] == scenario) & (uncertainty_df["type"] == output_name)]
        median_df = df[df["quantile"] == .5]
        time = [datetime.datetime.strptime(s, "%Y-%m-%d") for s in median_df['time']]
        
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
    plt.tight_layout()


def plot_final_size_compare(axis, uncertainty_df, output_name):
    update_rcparams()
    # plt.rcParams.update({'font.size': 12})    
    box_width = .7
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

    axis.set_xlim((0., 3.))
    axis.set_ylim((0, y_max * 1.2))
