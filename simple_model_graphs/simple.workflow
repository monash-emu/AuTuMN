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
		detect_smearpos
		missed_smearpos
		treatment_infect_smearpos
		treatment_noninfect_smearpos
		active_smearneg
		detect_smearneg
		missed_smearneg
		treatment_infect_smearneg
		treatment_noninfect_smearneg
		active_extrapul
		detect_extrapul
		missed_extrapul
		treatment_infect_extrapul
		treatment_noninfect_extrapul
		tb_death
			susceptible_fully -> latent_early [label=rate_force]
			susceptible_vac -> latent_early [label=rate_force_weak]
			susceptible_treated -> latent_early [label=rate_force_weak]
			latent_late -> latent_early [label=rate_force_weak]
			latent_early -> latent_late [label=0.8]
			latent_early -> active_smearpos [label=0.12]
			latent_late -> active_smearpos [label=0.000300]
			active_smearpos -> latent_late [label=0.15]
			active_smearpos -> detect_smearpos [label=0.8]
			active_smearpos -> missed_smearpos [label=0.20]
			detect_smearpos -> treatment_infect_smearpos [label=26.0]
			missed_smearpos -> active_smearpos [label=4.0]
			treatment_infect_smearpos -> treatment_noninfect_smearpos [label=23.4]
			treatment_infect_smearpos -> active_smearpos [label=1.3]
			treatment_noninfect_smearpos -> active_smearpos [label=0.20]
			treatment_noninfect_smearpos -> susceptible_treated [label=1.4]
			latent_early -> active_smearneg [label=0.0400]
			latent_late -> active_smearneg [label=0.000100]
			active_smearneg -> latent_late [label=0.15]
			active_smearneg -> detect_smearneg [label=0.8]
			active_smearneg -> missed_smearneg [label=0.20]
			detect_smearneg -> treatment_infect_smearneg [label=26.0]
			missed_smearneg -> active_smearneg [label=4.0]
			treatment_infect_smearneg -> treatment_noninfect_smearneg [label=23.4]
			treatment_infect_smearneg -> active_smearneg [label=1.3]
			treatment_noninfect_smearneg -> active_smearneg [label=0.20]
			treatment_noninfect_smearneg -> susceptible_treated [label=1.4]
			latent_early -> active_extrapul [label=0.0400]
			latent_late -> active_extrapul [label=0.000100]
			active_extrapul -> latent_late [label=0.15]
			active_extrapul -> detect_extrapul [label=0.8]
			active_extrapul -> missed_extrapul [label=0.20]
			detect_extrapul -> treatment_infect_extrapul [label=26.0]
			missed_extrapul -> active_extrapul [label=4.0]
			treatment_infect_extrapul -> treatment_noninfect_extrapul [label=23.4]
			treatment_infect_extrapul -> active_extrapul [label=1.3]
			treatment_noninfect_extrapul -> active_extrapul [label=0.20]
			treatment_noninfect_extrapul -> susceptible_treated [label=1.4]
			active_smearpos -> tb_death [label=0.15]
			detect_smearpos -> tb_death [label=0.15]
			treatment_infect_smearpos -> tb_death [label=1.3]
			treatment_noninfect_smearpos -> tb_death [label=0.20]
			active_smearneg -> tb_death [label=0.15]
			detect_smearneg -> tb_death [label=0.15]
			treatment_infect_smearneg -> tb_death [label=1.3]
			treatment_noninfect_smearneg -> tb_death [label=0.20]
			active_extrapul -> tb_death [label=0.15]
			detect_extrapul -> tb_death [label=0.15]
			treatment_infect_extrapul -> tb_death [label=1.3]
			treatment_noninfect_extrapul -> tb_death [label=0.20]
}