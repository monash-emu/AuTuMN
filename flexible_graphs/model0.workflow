digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully_nocomorbs
		susceptible_vac_nocomorbs
		susceptible_treated_nocomorbs
		latent_early_ds_nocomorbs
		latent_early_mdr_nocomorbs
		latent_late_ds_nocomorbs
		latent_late_mdr_nocomorbs
		active_smearpos_ds_nocomorbs
		active_smearneg_ds_nocomorbs
		active_extrapul_ds_nocomorbs
		active_smearpos_mdr_nocomorbs
		active_smearneg_mdr_nocomorbs
		active_extrapul_mdr_nocomorbs
		detect_smearpos_ds_nocomorbs
		detect_smearneg_ds_nocomorbs
		detect_extrapul_ds_nocomorbs
		detect_smearpos_mdr_nocomorbs
		detect_smearneg_mdr_nocomorbs
		detect_extrapul_mdr_nocomorbs
		missed_smearpos_ds_nocomorbs
		missed_smearneg_ds_nocomorbs
		missed_extrapul_ds_nocomorbs
		missed_smearpos_mdr_nocomorbs
		missed_smearneg_mdr_nocomorbs
		missed_extrapul_mdr_nocomorbs
		treatment_infect_smearpos_ds_nocomorbs
		treatment_infect_smearneg_ds_nocomorbs
		treatment_infect_extrapul_ds_nocomorbs
		treatment_infect_smearpos_mdr_nocomorbs
		treatment_infect_smearneg_mdr_nocomorbs
		treatment_infect_extrapul_mdr_nocomorbs
		treatment_noninfect_smearpos_ds_nocomorbs
		treatment_noninfect_smearneg_ds_nocomorbs
		treatment_noninfect_extrapul_ds_nocomorbs
		treatment_noninfect_smearpos_mdr_nocomorbs
		treatment_noninfect_smearneg_mdr_nocomorbs
		treatment_noninfect_extrapul_mdr_nocomorbs
		tb_death
			susceptible_fully_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_ds]
			susceptible_vac_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_treated_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			latent_late_ds_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_fully_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_mdr]
			susceptible_vac_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			susceptible_treated_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			latent_late_mdr_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			active_smearpos_ds_nocomorbs -> detect_smearpos_ds_nocomorbs [label=program_rate_detect]
			active_smearpos_ds_nocomorbs -> missed_smearpos_ds_nocomorbs [label=program_rate_missed]
			active_smearneg_ds_nocomorbs -> detect_smearneg_ds_nocomorbs [label=program_rate_detect]
			active_smearneg_ds_nocomorbs -> missed_smearneg_ds_nocomorbs [label=program_rate_missed]
			active_extrapul_ds_nocomorbs -> detect_extrapul_ds_nocomorbs [label=program_rate_detect]
			active_extrapul_ds_nocomorbs -> missed_extrapul_ds_nocomorbs [label=program_rate_missed]
			active_smearpos_mdr_nocomorbs -> detect_smearpos_mdr_nocomorbs [label=program_rate_detect]
			active_smearpos_mdr_nocomorbs -> missed_smearpos_mdr_nocomorbs [label=program_rate_missed]
			active_smearneg_mdr_nocomorbs -> detect_smearneg_mdr_nocomorbs [label=program_rate_detect]
			active_smearneg_mdr_nocomorbs -> missed_smearneg_mdr_nocomorbs [label=program_rate_missed]
			active_extrapul_mdr_nocomorbs -> detect_extrapul_mdr_nocomorbs [label=program_rate_detect]
			active_extrapul_mdr_nocomorbs -> missed_extrapul_mdr_nocomorbs [label=program_rate_missed]
			treatment_infect_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_smearpos_ds_nocomorbs -> active_smearpos_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_ds]
			treatment_infect_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_smearneg_ds_nocomorbs -> active_smearneg_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_ds]
			treatment_infect_extrapul_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=program_rate_default_noninfect_noamplify_ds]
			treatment_infect_extrapul_ds_nocomorbs -> active_extrapul_mdr_nocomorbs [label=program_rate_default_noninfect_amplify_ds]
			latent_early_ds_nocomorbs -> latent_late_ds_nocomorbs [label=2.2]
			latent_early_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.12]
			latent_late_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.0029]
			active_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.08]
			missed_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.08]
			detect_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.08]
			latent_early_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.17]
			latent_late_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.0040]
			active_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			missed_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			detect_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			latent_early_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=0.0051]
			latent_late_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=0.000120]
			active_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			missed_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			detect_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			latent_early_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=2.2]
			latent_early_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.12]
			latent_late_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.0029]
			active_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.08]
			missed_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.08]
			detect_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.08]
			latent_early_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.17]
			latent_late_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.0040]
			active_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			missed_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			detect_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			latent_early_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.0051]
			latent_late_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.000120]
			active_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			missed_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			detect_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			detect_smearpos_ds_nocomorbs -> treatment_infect_smearpos_ds_nocomorbs [label=26.0]
			missed_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=4.0]
			detect_smearneg_ds_nocomorbs -> treatment_infect_smearneg_ds_nocomorbs [label=26.0]
			missed_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=4.0]
			detect_extrapul_ds_nocomorbs -> treatment_infect_extrapul_ds_nocomorbs [label=26.0]
			missed_extrapul_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=4.0]
			detect_smearpos_mdr_nocomorbs -> treatment_infect_smearpos_mdr_nocomorbs [label=26.0]
			missed_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=4.0]
			detect_smearneg_mdr_nocomorbs -> treatment_infect_smearneg_mdr_nocomorbs [label=26.0]
			missed_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=4.0]
			detect_extrapul_mdr_nocomorbs -> treatment_infect_extrapul_mdr_nocomorbs [label=26.0]
			missed_extrapul_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=4.0]
			treatment_infect_smearpos_ds_nocomorbs -> treatment_noninfect_smearpos_ds_nocomorbs [label=28.4]
			treatment_noninfect_smearpos_ds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_smearpos_mdr_nocomorbs -> treatment_noninfect_smearpos_mdr_nocomorbs [label=11.8]
			treatment_noninfect_smearpos_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.18]
			treatment_noninfect_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.15]
			treatment_infect_smearneg_ds_nocomorbs -> treatment_noninfect_smearneg_ds_nocomorbs [label=28.4]
			treatment_noninfect_smearneg_ds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_smearneg_mdr_nocomorbs -> treatment_noninfect_smearneg_mdr_nocomorbs [label=11.8]
			treatment_noninfect_smearneg_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.18]
			treatment_noninfect_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.15]
			treatment_infect_extrapul_ds_nocomorbs -> treatment_noninfect_extrapul_ds_nocomorbs [label=28.4]
			treatment_noninfect_extrapul_ds_nocomorbs -> susceptible_treated_nocomorbs [label=2.0]
			treatment_infect_extrapul_mdr_nocomorbs -> treatment_noninfect_extrapul_mdr_nocomorbs [label=11.8]
			treatment_noninfect_extrapul_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=0.32]
			treatment_infect_extrapul_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.18]
			treatment_noninfect_extrapul_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.15]
			active_smearpos_ds_nocomorbs -> tb_death [label=0.17]
			missed_smearpos_ds_nocomorbs -> tb_death [label=0.17]
			detect_smearpos_ds_nocomorbs -> tb_death [label=0.17]
			active_smearneg_ds_nocomorbs -> tb_death [label=0.0500]
			missed_smearneg_ds_nocomorbs -> tb_death [label=0.0500]
			detect_smearneg_ds_nocomorbs -> tb_death [label=0.0500]
			active_extrapul_ds_nocomorbs -> tb_death [label=0.0500]
			missed_extrapul_ds_nocomorbs -> tb_death [label=0.0500]
			detect_extrapul_ds_nocomorbs -> tb_death [label=0.0500]
			active_smearpos_mdr_nocomorbs -> tb_death [label=0.17]
			missed_smearpos_mdr_nocomorbs -> tb_death [label=0.17]
			detect_smearpos_mdr_nocomorbs -> tb_death [label=0.17]
			active_smearneg_mdr_nocomorbs -> tb_death [label=0.0500]
			missed_smearneg_mdr_nocomorbs -> tb_death [label=0.0500]
			detect_smearneg_mdr_nocomorbs -> tb_death [label=0.0500]
			active_extrapul_mdr_nocomorbs -> tb_death [label=0.0500]
			missed_extrapul_mdr_nocomorbs -> tb_death [label=0.0500]
			detect_extrapul_mdr_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_smearpos_ds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_ds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_smearpos_mdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearpos_mdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_smearneg_ds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearneg_ds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_smearneg_mdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_smearneg_mdr_nocomorbs -> tb_death [label=0.0499]
			treatment_infect_extrapul_ds_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_extrapul_ds_nocomorbs -> tb_death [label=0.0500]
			treatment_infect_extrapul_mdr_nocomorbs -> tb_death [label=0.05]
			treatment_noninfect_extrapul_mdr_nocomorbs -> tb_death [label=0.0499]
}