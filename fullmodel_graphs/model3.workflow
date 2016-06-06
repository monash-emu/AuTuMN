digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully
		susceptible_vac
		susceptible_treated
		latent_early
		latent_late
		active_smearpos
		active_smearneg
		active_extrapul
		detect_smearpos
		detect_smearneg
		detect_extrapul
		missed_smearpos
		missed_smearneg
		missed_extrapul
		treatment_infect_smearpos
		treatment_infect_smearneg
		treatment_infect_extrapul
		treatment_noninfect_smearpos
		treatment_noninfect_smearneg
		treatment_noninfect_extrapul
		tb_death
			susceptible_fully -> latent_early [label=rate]
			susceptible_vac -> latent_early [label=rate]
			susceptible_treated -> latent_early [label=rate]
			latent_late -> latent_early [label=rate]
			latent_early -> active_smearpos [label=tb_r]
			latent_late -> active_smearpos [label=tb_r]
			latent_early -> active_smearneg [label=tb_r]
			latent_late -> active_smearneg [label=tb_r]
			latent_early -> active_extrapul [label=tb_r]
			latent_late -> active_extrapul [label=tb_r]
			active_smearpos -> missed_smearpos [label=prog]
			active_smearneg -> missed_smearneg [label=prog]
			active_extrapul -> missed_extrapul [label=prog]
			active_smearpos -> detect_smearpos [label=prog]
			active_smearneg -> detect_smearneg [label=prog]
			active_extrapul -> detect_extrapul [label=prog]
			treatment_infect_smearpos -> treatment_noninfect_smearpos [label=prog]
			treatment_noninfect_smearpos -> susceptible_treated [label=prog]
			treatment_infect_smearpos -> active_smearpos [label=prog]
			treatment_noninfect_smearpos -> active_smearpos [label=prog]
			treatment_infect_smearneg -> treatment_noninfect_smearneg [label=prog]
			treatment_noninfect_smearneg -> susceptible_treated [label=prog]
			treatment_infect_smearneg -> active_smearneg [label=prog]
			treatment_noninfect_smearneg -> active_smearneg [label=prog]
			treatment_infect_extrapul -> treatment_noninfect_extrapul [label=prog]
			treatment_noninfect_extrapul -> susceptible_treated [label=prog]
			treatment_infect_extrapul -> active_extrapul [label=prog]
			treatment_noninfect_extrapul -> active_extrapul [label=prog]
			latent_early -> latent_late [label=5.5]
			active_smearpos -> latent_late [label=0.10]
			missed_smearpos -> latent_late [label=0.10]
			detect_smearpos -> latent_late [label=0.10]
			active_smearneg -> latent_late [label=0.27]
			missed_smearneg -> latent_late [label=0.27]
			detect_smearneg -> latent_late [label=0.27]
			active_extrapul -> latent_late [label=0.27]
			missed_extrapul -> latent_late [label=0.27]
			detect_extrapul -> latent_late [label=0.27]
			missed_smearpos -> active_smearpos [label=4.0]
			missed_smearneg -> active_smearneg [label=4.0]
			missed_extrapul -> active_extrapul [label=4.0]
			detect_smearpos -> treatment_infect_smearpos [label=26.0]
			detect_smearneg -> treatment_infect_smearneg [label=26.0]
			detect_extrapul -> treatment_infect_extrapul [label=26.0]
			active_smearpos -> tb_death [label=0.23]
			missed_smearpos -> tb_death [label=0.23]
			detect_smearpos -> tb_death [label=0.23]
			active_smearneg -> tb_death [label=0.07]
			missed_smearneg -> tb_death [label=0.07]
			detect_smearneg -> tb_death [label=0.07]
			active_extrapul -> tb_death [label=0.07]
			missed_extrapul -> tb_death [label=0.07]
			detect_extrapul -> tb_death [label=0.07]
			treatment_infect_smearpos -> tb_death [label=prog]
			treatment_noninfect_smearpos -> tb_death [label=prog]
			treatment_infect_smearneg -> tb_death [label=prog]
			treatment_noninfect_smearneg -> tb_death [label=prog]
			treatment_infect_extrapul -> tb_death [label=prog]
			treatment_noninfect_extrapul -> tb_death [label=prog]
}