digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully_remote
		susceptible_vac_remote
		susceptible_treated_remote
		latent_early_remote
		latent_late_remote
		active_remote
		detect_remote
		missed_remote
		treatment_infect_remote
		treatment_noninfect_remote
		susceptible_fully_prison
		susceptible_vac_prison
		susceptible_treated_prison
		latent_early_prison
		latent_late_prison
		active_prison
		detect_prison
		missed_prison
		treatment_infect_prison
		treatment_noninfect_prison
		susceptible_fully_nocomorb
		susceptible_vac_nocomorb
		susceptible_treated_nocomorb
		latent_early_nocomorb
		latent_late_nocomorb
		active_nocomorb
		detect_nocomorb
		missed_nocomorb
		treatment_infect_nocomorb
		treatment_noninfect_nocomorb
		tb_death
			susceptible_fully_remote -> latent_early_remote [label=rate]
			susceptible_vac_remote -> latent_early_remote [label=rate]
			susceptible_treated_remote -> latent_early_remote [label=rate]
			latent_late_remote -> latent_early_remote [label=rate]
			susceptible_fully_prison -> latent_early_prison [label=rate]
			susceptible_vac_prison -> latent_early_prison [label=rate]
			susceptible_treated_prison -> latent_early_prison [label=rate]
			latent_late_prison -> latent_early_prison [label=rate]
			susceptible_fully_nocomorb -> latent_early_nocomorb [label=rate]
			susceptible_vac_nocomorb -> latent_early_nocomorb [label=rate]
			susceptible_treated_nocomorb -> latent_early_nocomorb [label=rate]
			latent_late_nocomorb -> latent_early_nocomorb [label=rate]
			active_remote -> missed_remote [label=prog]
			detect_remote -> treatment_infect_remote [label=prog]
			active_prison -> missed_prison [label=prog]
			detect_prison -> treatment_infect_prison [label=prog]
			active_nocomorb -> missed_nocomorb [label=prog]
			detect_nocomorb -> treatment_infect_nocomorb [label=prog]
			active_remote -> detect_remote [label=prog]
			active_prison -> detect_prison [label=prog]
			active_nocomorb -> detect_nocomorb [label=prog]
			treatment_infect_remote -> treatment_noninfect_remote [label=prog]
			treatment_noninfect_remote -> susceptible_treated_remote [label=prog]
			treatment_infect_remote -> active_remote [label=prog]
			treatment_noninfect_remote -> active_remote [label=prog]
			treatment_infect_prison -> treatment_noninfect_prison [label=prog]
			treatment_noninfect_prison -> susceptible_treated_prison [label=prog]
			treatment_infect_prison -> active_prison [label=prog]
			treatment_noninfect_prison -> active_prison [label=prog]
			treatment_infect_nocomorb -> treatment_noninfect_nocomorb [label=prog]
			treatment_noninfect_nocomorb -> susceptible_treated_nocomorb [label=prog]
			treatment_infect_nocomorb -> active_nocomorb [label=prog]
			treatment_noninfect_nocomorb -> active_nocomorb [label=prog]
			latent_early_remote -> latent_late_remote [label=5.3]
			latent_early_remote -> active_remote [label=0.8]
			latent_late_remote -> active_remote [label=0.0025]
			latent_early_prison -> latent_late_prison [label=5.3]
			latent_early_prison -> active_prison [label=0.8]
			latent_late_prison -> active_prison [label=0.0025]
			latent_early_nocomorb -> latent_late_nocomorb [label=5.3]
			latent_early_nocomorb -> active_nocomorb [label=0.8]
			latent_late_nocomorb -> active_nocomorb [label=0.0025]
			active_remote -> latent_late_remote [label=0.20]
			missed_remote -> latent_late_remote [label=0.20]
			detect_remote -> latent_late_remote [label=0.20]
			active_prison -> latent_late_prison [label=0.20]
			missed_prison -> latent_late_prison [label=0.20]
			detect_prison -> latent_late_prison [label=0.20]
			active_nocomorb -> latent_late_nocomorb [label=0.20]
			missed_nocomorb -> latent_late_nocomorb [label=0.20]
			detect_nocomorb -> latent_late_nocomorb [label=0.20]
			missed_remote -> active_remote [label=4.0]
			missed_prison -> active_prison [label=4.0]
			missed_nocomorb -> active_nocomorb [label=4.0]
			latent_early_remote -> susceptible_vac_remote [label=link]
			latent_early_prison -> susceptible_vac_prison [label=link]
			latent_early_nocomorb -> susceptible_vac_nocomorb [label=link]
			active_remote -> tb_death [label=0.13]
			missed_remote -> tb_death [label=0.13]
			detect_remote -> tb_death [label=0.13]
			active_prison -> tb_death [label=0.13]
			missed_prison -> tb_death [label=0.13]
			detect_prison -> tb_death [label=0.13]
			active_nocomorb -> tb_death [label=0.13]
			missed_nocomorb -> tb_death [label=0.13]
			detect_nocomorb -> tb_death [label=0.13]
			treatment_infect_remote -> tb_death [label=prog]
			treatment_noninfect_remote -> tb_death [label=prog]
			treatment_infect_prison -> tb_death [label=prog]
			treatment_noninfect_prison -> tb_death [label=prog]
			treatment_infect_nocomorb -> tb_death [label=prog]
			treatment_noninfect_nocomorb -> tb_death [label=prog]
}