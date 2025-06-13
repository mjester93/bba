import streamlit as st
import pandas as pd

from helpers import co_draft_multiple, next_pick_distribution


# Cache data loading for performance
def load_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def main() -> None:
    st.title("Best Ball Co-Draft Explorer")
    st.markdown(
        "Upload your draft CSV and select players or a single player to explore draft patterns."
    )

    uploaded_file = st.file_uploader("Upload draft data (CSV)", type=["csv"])
    if not uploaded_file:
        st.info("🔄 Please upload a CSV file to get started.")
        return

    df = load_data(uploaded_file)
    player_list = sorted(df["PLAYERNAME"].unique())

    # Co-draft multi-player analysis
    st.sidebar.header("Co-Draft Analysis")
    selected_players = st.sidebar.multiselect(
        "Select players to analyze co-draft:", player_list
    )
    if st.sidebar.button("Compute Co-Draft Stats"):
        if not selected_players:
            st.sidebar.error("Select at least one player.")
        else:
            stats = co_draft_multiple(df, selected_players)
            st.subheader("Co-Draft Results")
            st.write(f"**Analyzed Players:** {', '.join(stats['players'])}")
            st.write(
                f"- Times {stats['players'][0]} was drafted: **{stats['num_first']}**"
            )
            st.write(
                f"- Times all selected were drafted together: **{stats['num_all']}**"
            )
            st.write(f"- Percentage together: **{stats['pct_together']:.1f}%**")

            if stats["common_pairs"]:
                st.subheader("Full Team Rosters for Co-Draft Occurrences")
                roster_records = []
                for draft_num, slot in stats["common_pairs"]:
                    team_players = df.loc[
                        (df["SNAKEDRAFTNUM"] == draft_num) & (df["draft_slot"] == slot),
                        "PLAYERNAME",
                    ].tolist()
                    roster_records.append(
                        {
                            "SNAKEDRAFTNUM": draft_num,
                            "draft_slot": slot,
                            "team_roster": ", ".join(team_players),
                        }
                    )
                roster_df = pd.DataFrame(roster_records)
                st.dataframe(roster_df)

    # Next-pick distribution for single player
    st.sidebar.header("Next-Pick Distribution")
    single_player = st.sidebar.selectbox("Select a player for next-pick:", player_list)
    if st.sidebar.button("Show Next-Pick Distribution"):
        st.subheader(f"Next-Pick Distribution after {single_player}")
        dist_df = next_pick_distribution(df, single_player)
        if dist_df.empty:
            st.write(f"No subsequent picks found for {single_player}.")
        else:
            st.dataframe(dist_df)


if __name__ == "__main__":
    main()
