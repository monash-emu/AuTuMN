from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from math import ceil

import plotly.graph_objects as go
import seaborn as sns

YLAB_LOOKUP_SPLIT = {
   "cases_averted_relative": "% infections averted<br>by school closure", 
   "deaths_averted_relative": "% deaths averted<br>by school closure",
   "delta_hospital_peak_relative": "Relative reduction in<br>peak hospital occupancy (%)"
}

BOX_COLORS= {
   "cases_averted_relative": "black",
   "deaths_averted_relative": "purple",
   "delta_hospital_peak_relative": "firebrick"
}

ANALYSIS_COLORS = {
    "main": "black",
    "increased_hh_contacts": "firebrick",
    "no_google_mobility": "mediumblue"
}

ANALYSIS_TITLES = {
   "increased_hh_contacts": "SA2: Increased household contacts during closures", 
   "no_google_mobility": "SA1: Excluding Google mobility data"
}

def plot_multic_relative_outputs(output_dfs_dict: dict[str, pd.DataFrame], req_outputs=["cases_averted_relative", "deaths_averted_relative", "delta_hospital_peak_relative"]):
    n_subplots = len(req_outputs)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(25, n_subplots*6))

    this_iso3_list = list(output_dfs_dict.keys())
    n_countries = len(this_iso3_list)

    box_width = .4
    med_color = 'white'
    
    for i_output, output in enumerate(req_outputs):
        box_color = BOX_COLORS[output]
        axis = axes[i_output]

        mean_values = [output_dfs_dict[iso3][output].loc[0.5] for iso3 in this_iso3_list]
        sorted_iso3_list = [iso3 for _ , iso3 in sorted(zip(mean_values, this_iso3_list))]
        
        y_max_abs = 0.
        for i_iso3, iso3 in enumerate(sorted_iso3_list):
            x = i_iso3 + 1

            data = - 100. * output_dfs_dict[iso3][output] # use %. And use "-" so positive nbs indicate positive effect of closures

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

            axis.set_xlim((0, n_countries + 1))
            axis.set_ylim(-1.2*y_max_abs, 1.2*y_max_abs)

        axis.set_xticks(ticks=range(1, n_countries + 1), labels=sorted_iso3_list, rotation=90, fontsize=13)

        y_label = YLAB_LOOKUP_SPLIT[output].replace("<br>", " ")
        axis.set_ylabel(y_label, fontsize=15)

        # add coloured backgorund patches
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim() 
        rect_up = Rectangle(xy=(xmin, 0.), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="white")  #"honeydew")
        axis.add_patch(rect_up)
        rect_low = Rectangle(xy=(xmin, ymin), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="gainsboro", alpha=.5)  #"mistyrose")
        axis.add_patch(rect_low)

        axis.text(n_countries * .75, ymax / 2., s="Positive effect of\nschool closures", fontsize=13)
        axis.text(n_countries * .25, ymin / 2., s="Negative effect of\nschool closures", fontsize=13)   

    plt.tight_layout()

    return fig


def plot_analyses_comparison(output_dfs_dict: dict[str, dict], output="cases_averted_relative"):

   sas = ['main', 'no_google_mobility', 'increased_hh_contacts']
   
   sa_short_title = {
      "main": "Base-case",
      "no_google_mobility": "SA1: No Google mobility",
      "increased_hh_contacts": "SA2: Increased hh contacts"
   }

   n_countries = len(output_dfs_dict)
   n_subplots = 3
   n_countries_per_subplot = ceil(n_countries / n_subplots)
   this_iso3_list = list(output_dfs_dict.keys())

   fig, axes = plt.subplots(n_subplots, 1, figsize=(25, n_subplots*6))

   box_width = .4
   med_color = 'white'

   for i_subplot in range(n_subplots):
      iso3_sublist = this_iso3_list[i_subplot * n_countries_per_subplot: (i_subplot + 1) * n_countries_per_subplot]

      axis = axes[i_subplot]
      
      y_max_abs = 0.
      x_ticks = []
      for i_iso3, iso3 in enumerate(iso3_sublist):
         x_ticks.append(1. + 3 * i_iso3 + 1)
         for i_analysis, analysis in enumerate(sas):
            box_color = ANALYSIS_COLORS[analysis]
            x = 1. + 3 * i_iso3 + i_analysis

            data = - 100. * output_dfs_dict[iso3][analysis][output] # use %. And use "-" so positive nbs indicate positive effect of closures

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
         
         axis.vlines(x=x+.5,ymin=-1.e6,ymax=1.e6, lw=1., color="black", zorder=5)

      axis.set_xlim((0, 3 * n_countries_per_subplot + 1))
      axis.set_ylim(-1.2*y_max_abs, 1.2*y_max_abs)

      axis.set_xticks(ticks=x_ticks, labels=iso3_sublist, rotation=90, fontsize=13)

      y_label = YLAB_LOOKUP_SPLIT[output].replace("<br>", " ")
      axis.set_ylabel(y_label, fontsize=15)

      # add pseudo-legend
      if i_subplot == 0:
         if output == "deaths_averted_relative":
            legend_y = 1.19 * y_max_abs
            va = "top"
         else:
            legend_y = -1.15 * y_max_abs
            va = "bottom"
         for i_analysis, analysis in enumerate(sas):
            axis.text(x=1. + i_analysis, y=legend_y, s=sa_short_title[analysis], fontsize=10, rotation=90, color=ANALYSIS_COLORS[analysis], ha="center", va=va)

      # add coloured backgorund patches
      xmin, xmax = axis.get_xlim()
      ymin, ymax = axis.get_ylim() 
      rect_up = Rectangle(xy=(xmin, 0.), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="white")  #"honeydew")
      axis.add_patch(rect_up)
      rect_low = Rectangle(xy=(xmin, ymin), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="gainsboro", alpha=.5)  #"mistyrose")
      axis.add_patch(rect_low)

   plt.tight_layout()

   return fig


def plot_analyses_median_deltas(diff_quantiles_dfs):
   this_iso3_list = list(diff_quantiles_dfs.keys())

   sas = ["no_google_mobility", "increased_hh_contacts"]
   median_deltas = {}
   for output in ["cases_averted_relative", "deaths_averted_relative", "delta_hospital_peak_relative"]:
      median_deltas[output] = {}
      for analysis in sas:
         deltas = [
               - 100. * (diff_quantiles_dfs[iso3][analysis][output][0.5] - diff_quantiles_dfs[iso3]["main"][output][0.5])
               for iso3 in this_iso3_list
         ]
         median_deltas[output][analysis] = deltas

   fig, axes = plt.subplots(3, 2, figsize=(12, 12))

   for i, output in enumerate(["cases_averted_relative", "deaths_averted_relative", "delta_hospital_peak_relative"]):
      for j, analysis in enumerate(sas):      
         ax = axes[i, j]
         pd.Series(median_deltas[output][analysis]).plot.hist(ax = ax, color=ANALYSIS_COLORS[analysis], alpha=.5)
         ymax = ax.get_ylim()[1]
         xmin, xmax = ax.get_xlim()
         ax.text(s="base-case", x=.015 * (xmax-xmin), y=ymax, rotation=90, ha="left", va="top", fontsize=12)

         ymax = ax.get_ylim()[1]
         ax.vlines(x=0, ymin=0, ymax=ymax, colors='black', lw=2, linestyles=["dashed"])
         ax.set_xlabel("Absolute difference between median estimates (%)")
         ax.set_ylabel("Frenquency (N countries)")

   # Add column labels

   pad=30
   for ax, sa in zip(axes[0], sas):
      ax.annotate(
         ANALYSIS_TITLES[sa], xy=(0.5, 1), xytext=(0, pad),
         xycoords='axes fraction', textcoords='offset points',
         size='x-large', ha='center', va='baseline')

   # Add row labels
   pad=40
   for ax, output in zip(axes[:, 0], ["cases_averted_relative", "deaths_averted_relative", "delta_hospital_peak_relative"]):
      ax.annotate(
         YLAB_LOOKUP_SPLIT[output].replace("<br>", "\n"), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
         xycoords=ax.yaxis.label, textcoords='offset points',
         size='x-large', ha='center', va='center', rotation=90)

   # Draw separation lines
   # vertical lines
   vertical_xs = [.07, .51]
   for x in vertical_xs:
      plt.plot([x, x], [0, 1.], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)
   # horizontal lines
   horizontal_ys = [.315, .63, .945]
   for y in horizontal_ys:
      plt.plot([0, 1], [y, y], color='black', lw=1, transform=plt.gcf().transFigure, clip_on=False)
   plt.tight_layout()
   return fig


def plot_ll_comparison(combined_ll_df, output="loglikelihood"):

   column_rename = {"ll_extra_ll": "Random process loglikelihood", "logposterior": "Posterior loglikelihood (with offset)"}

   combined_ll_df.replace("main", "Base-case", inplace=True)
   combined_ll_df.replace("increased_hh_contacts", "Increased hh contacts", inplace=True)
   combined_ll_df.replace("no_google_mobility", "No Google mobility data", inplace=True)

   combined_ll_df.rename(columns=column_rename, inplace=True)

   this_iso3_list = combined_ll_df.iso3.unique()

   n_countries = len(this_iso3_list)
   n_subplots = 3
   n_countries_per_subplot = ceil(n_countries / n_subplots)

   fig, axes = plt.subplots(n_subplots, 1, figsize=(25, n_subplots*6))

   for i_subplot in range(n_subplots):
      iso3_sublist = this_iso3_list[i_subplot * n_countries_per_subplot: (i_subplot + 1) * n_countries_per_subplot]

      this_df = combined_ll_df[combined_ll_df['iso3'].isin(iso3_sublist)]
      axis = axes[i_subplot]      

      sns.violinplot(this_df, x="iso3", y=column_rename[output], hue="analysis", ax=axis)

      locs = axis.get_xticks()
      vlines_xs = [l + .5 for l in locs[:-1]]
      ymin, ymax = axis.get_ylim()
      axis.vlines(x=vlines_xs, ymin=ymin, ymax=ymax, colors="black", lw=.5)
      axis.set_xlim((-.5, n_countries_per_subplot - .5))
      
      axis.yaxis.label.set_size(15)

   plt.tight_layout()

   return fig


def plot_relative_map(output_dfs_dict: dict[str, pd.DataFrame], req_output="delta_hospital_peak_relative"):
   this_iso3_list = list(output_dfs_dict.keys())
   values = [- 100 * output_dfs_dict[iso3][req_output].loc[0.5] for iso3 in this_iso3_list]
   data_df = pd.DataFrame.from_dict({"iso3": this_iso3_list, "values": values})
   
   legend_title = YLAB_LOOKUP_SPLIT[req_output]

   fig = go.Figure(
      data=go.Choropleth(
         locations=data_df["iso3"], 
         z=data_df["values"],
         colorscale= [[0, 'lightblue'],
                  #   [0.5, 'darkgrey'],
                    [1, 'blue']],  # "Plotly3",
         marker_line_color='darkgrey',
         marker_line_width=0.5,
         colorbar_title=legend_title,
         colorbar_ticksuffix="%",
      )  
   )
   fig.update_layout(
      geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'       #'natural earth'   'equirectangular'
         ),
      margin={"r":0,"t":0,"l":0,"b":0},
      autosize=False,
      height=500,
      width=1200,
   )
   return fig