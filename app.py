"""Auction Bidding Strategy Analyzer  (GTDS mini project)
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import auction_engine as ae

st.set_page_config(page_title="Auction Bidding Strategy Analyzer", layout="wide")
st.title("Auction Bidding Strategy Analyzer")
st.caption("Game Theory in Data Science: first-price vs second-price (Vickrey) auctions")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Global settings")
    reserve = st.number_input("Reserve price (₹)", min_value=0.0, value=0.0, step=100.0,
                              help="Seller will not sell below this price. 0 = no reserve.")
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.markdown("Currency is shown in ₹. In the Monte Carlo tabs values are on a 0-100 scale "
                "(think of it as ₹ thousands).")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Single auction", "2. Repeated experiments", "3. Strategy lab", "4. Game theory notes"]
)

# ---------------- Tab 1: single auction ----------------
with tab1:
    st.subheader("Set up bidders")
    default = pd.DataFrame({
        "Bidder": ["A", "B", "C"],
        "Valuation (₹)": [10000.0, 8000.0, 6000.0],
        "Strategy": ["Truthful"] * 3,
        "Adjust % (for shade/overbid)": [10.0, 10.0, 10.0],
    })
    edited = st.data_editor(
        default, num_rows="dynamic", width="stretch", key="bidders",
        column_config={
            "Strategy": st.column_config.SelectboxColumn(options=ae.STRATEGIES, required=True),
            "Valuation (₹)": st.column_config.NumberColumn(min_value=0.0, step=100.0),
            "Adjust % (for shade/overbid)": st.column_config.NumberColumn(min_value=0.0, max_value=90.0),
        },
    ).dropna()

    n = len(edited)
    if n < 2 or n > 10:
        st.warning("Please keep between 2 and 10 bidders.")
    else:
        vals = edited["Valuation (₹)"].to_numpy(float)
        facs = np.array([ae.bid_factor(s, p, n) for s, p in
                         zip(edited["Strategy"], edited["Adjust % (for shade/overbid)"])])
        bids = vals * facs
        names = edited["Bidder"].astype(str).tolist()

        rows, cols = [], st.columns(2)
        for mech, col in zip(ae.MECHANISMS, cols):
            res = ae.run_auctions(vals, bids, mech, reserve, np.random.default_rng(int(seed)))
            w = int(res["winner"][0])
            with col:
                st.markdown(f"#### {mech}")
                if w < 0:
                    st.error("No sale: highest bid is below the reserve price.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Winner", names[w])
                    c2.metric("Winner pays", f"₹{res['price'][0]:,.0f}")
                    c3.metric("Winner profit", f"₹{res['profits'][0, w]:,.0f}")
                    st.metric("Seller revenue", f"₹{res['revenue'][0]:,.0f}")
                    st.caption("Efficient allocation (highest valuation won)"
                               if res["efficient"][0] else
                               "Inefficient: the highest-value bidder did NOT win.")
            rows.append(pd.DataFrame({"Bidder": names, "Mechanism": mech,
                                      "Valuation": vals, "Bid": bids,
                                      "Profit": res["profits"][0]}))

        detail = pd.concat(rows)
        fig = go.Figure()
        fig.add_bar(x=names, y=vals, name="Valuation")
        fig.add_bar(x=names, y=bids, name="Submitted bid")
        fig.update_layout(barmode="group", title="Valuation vs submitted bid",
                          yaxis_title="₹", height=350)
        st.plotly_chart(fig, width="stretch")

        fig2 = px.bar(detail, x="Bidder", y="Profit", color="Mechanism", barmode="group",
                      title="Bidder profit by mechanism (profit = valuation - price paid, 0 if lose)")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, width="stretch")
        st.caption("Tip: set every bidder to 'Truthful' and both auctions pick the same winner, "
                   "but the seller earns more under first-price. Then try shading in the "
                   "second-price auction and see that it can only hurt you.")

# ---------------- Tab 2: Monte Carlo ----------------
with tab2:
    st.subheader("Run hundreds of random auctions")
    c1, c2, c3, c4 = st.columns(4)
    n_b = c1.slider("Number of bidders", 2, 10, 5)
    runs = c2.slider("Number of auctions", 100, 20000, 2000, step=100)
    dist = c3.selectbox("Valuation distribution", ae.DISTRIBUTIONS)
    play = c4.selectbox("How do bidders behave?", [
        "Equilibrium play (shade in first-price, truthful in second-price)",
        "Everyone bids truthfully",
        "Everyone shades their bid by a fixed %",
    ])
    shade = 15
    if play.startswith("Everyone shades"):
        shade = st.slider("Shading %", 0, 60, 15)

    def factor_for(mech, n_):
        if play.startswith("Equilibrium"):
            return (n_ - 1) / n_ if mech == ae.FIRST else 1.0
        if play.startswith("Everyone bids"):
            return 1.0
        return 1 - shade / 100

    rng = np.random.default_rng(int(seed))
    values = ae.draw_values(rng, runs, n_b, dist)

    summ, results = {}, {}
    for mech in ae.MECHANISMS:
        results[mech] = ae.monte_carlo(values, factor_for(mech, n_b), mech, reserve, int(seed))
        summ[mech] = ae.summarize(results[mech])
    table = pd.DataFrame(summ).T.round(2)
    st.dataframe(table, width="stretch")

    colA, colB = st.columns(2)
    with colA:
        hist = pd.concat([pd.DataFrame({"Revenue": results[m]["revenue"], "Mechanism": m})
                          for m in ae.MECHANISMS])
        fig = px.histogram(hist, x="Revenue", color="Mechanism", barmode="overlay",
                           opacity=0.6, nbins=50, title="Seller revenue distribution")
        st.plotly_chart(fig, width="stretch")
    with colB:
        prof = pd.DataFrame({m: results[m]["profits"].sum(axis=1) for m in ae.MECHANISMS})
        fig = px.box(prof.melt(var_name="Mechanism", value_name="Winner profit"),
                     x="Mechanism", y="Winner profit", title="Winner profit per auction")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Effect of the number of bidders")
    sweep = []
    for k in range(2, 11):
        vk = ae.draw_values(np.random.default_rng(int(seed) + k), runs, k, dist)
        for mech in ae.MECHANISMS:
            r = ae.monte_carlo(vk, factor_for(mech, k), mech, reserve, int(seed))
            sweep.append({"Bidders": k, "Mechanism": mech,
                          "Avg revenue": r["revenue"].mean(),
                          "Avg winner profit": r["profits"].sum(axis=1).mean(),
                          "Efficiency %": 100 * r["efficient"].mean()})
    sw = pd.DataFrame(sweep)
    m1, m2, m3 = st.columns(3)
    for col, y in zip((m1, m2, m3), ("Avg revenue", "Avg winner profit", "Efficiency %")):
        col.plotly_chart(px.line(sw, x="Bidders", y=y, color="Mechanism", markers=True, title=y),
                         width="stretch")

    st.info("Under equilibrium play with uniform values, both auctions yield (almost) the same "
            "average revenue (the Revenue Equivalence Theorem). If bidders are not at "
            "equilibrium (e.g. everyone shades a fixed %), revenue differs, and neither "
            "mechanism wins in every case.")

# ---------------- Tab 3: strategy lab ----------------
with tab3:
    st.subheader("Is truthful bidding the best strategy?")
    st.write("Bidder 1 tries many bid levels (as a fraction of their valuation) while all "
             "other bidders follow a fixed strategy. We measure bidder 1's average profit.")
    c1, c2, c3 = st.columns(3)
    n_l = c1.slider("Number of bidders ", 2, 10, 4, key="nl")
    runs_l = c2.slider("Number of auctions ", 500, 20000, 5000, step=500, key="rl")
    opp = c3.selectbox("Opponents' strategy", ["Truthful", "Equilibrium shading"])
    opp_f = ae.bid_factor(opp, 0, n_l)

    vals_l = ae.draw_values(np.random.default_rng(int(seed)), runs_l, n_l, "Uniform (0-100)")
    grid = np.round(np.arange(0.4, 1.3001, 0.02), 2)
    rows = []
    for mech in ae.MECHANISMS:
        for f, prof, pw in ae.focal_sweep(vals_l, grid, opp_f, mech, reserve, int(seed)):
            rows.append({"Bid factor": f, "Mechanism": mech, "Avg profit": prof, "Win prob": pw})
    df = pd.DataFrame(rows)

    fig = px.line(df, x="Bid factor", y="Avg profit", color="Mechanism",
                  title="Bidder 1 average profit vs bid / valuation")
    fig.add_vline(x=1.0, line_dash="dot", annotation_text="truthful")
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(px.line(df, x="Bid factor", y="Win prob", color="Mechanism",
                            title="Probability of winning"), width="stretch")

    best = df.loc[df.groupby("Mechanism")["Avg profit"].idxmax()]
    for _, r in best.iterrows():
        st.write(f"**{r['Mechanism']}**: best bid factor ≈ **{r['Bid factor']:.2f}** "
                 f"(avg profit {r['Avg profit']:.2f}).")
    st.caption(f"Theory for uniform values: second-price optimum = 1.00 (dominant strategy); "
               f"first-price best response = (n-1)/n = {(n_l-1)/n_l:.2f}. "
               "Small differences come from simulation noise.")

    st.markdown("#### Named strategy comparison (bidder 1)")
    named = [("Truthful", 0), ("Shade (bid below value)", 10), ("Shade (bid below value)", 25),
             ("Equilibrium shading", 0), ("Overbid (bid above value)", 10)]
    rows = []
    for s, p in named:
        f = ae.bid_factor(s, p, n_l)
        row = {"Strategy": f"{s} {p}%" if p else s, "Bid factor": round(f, 2)}
        for mech in ae.MECHANISMS:
            _, prof, pw = ae.focal_sweep(vals_l, [f], opp_f, mech, reserve, int(seed))[0]
            row[f"{mech} profit"] = round(prof, 3)
            row[f"{mech} win %"] = round(100 * pw, 1)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch")

# ---------------- Tab 4: notes ----------------
with tab4:
    st.markdown("""
### Game theory behind the simulator

**Payoff.** A bidder's payoff is `valuation - price paid` if they win, and `0` if they lose.

**First-price auction.** The highest bid wins and pays their own bid. Bidding your true
value guarantees zero profit, so rational bidders *shade* their bids. Truthful bidding is
**not** a dominant strategy here. With *n* bidders and i.i.d. uniform values, the symmetric
Bayes-Nash equilibrium is `bid = value × (n-1)/n`.

**Second-price (Vickrey) auction.** The highest bid wins but pays the second-highest bid.
Your bid only decides *whether* you win, not *what you pay*, so bidding your true value is a
**weakly dominant strategy** (private values, no collusion).

**Revenue Equivalence Theorem.** With risk-neutral bidders, independent private values and
equilibrium play, both auctions give the seller the same expected revenue. In practice,
revenue can differ when bidders are not at equilibrium, when values are correlated, or with
reserve prices.

**Allocation efficiency.** Share of auctions where the bidder with the highest valuation
wins. Truthful second-price bidding is always efficient. First-price with identical
shading is also efficient; mixed strategies can make it inefficient.

**Assumptions.** Seller's value is 0, private values, risk-neutral bidders, ties broken at
random. "Equilibrium shading" is exact only for uniform values and no reserve price.
""")
