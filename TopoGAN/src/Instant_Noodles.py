import pandas as pd
in_directory = "Results/Evaluation Results"

aggregate_topo_error = [[32, 5.850293732],[0, 6.030745983]]
aggregate_topo_error = pd.DataFrame(data=aggregate_topo_error, columns=["Seed", "Mean Error"]).sort_values(
        by="Mean Error", ascending=True)

print(aggregate_topo_error)
print(aggregate_topo_error.head(1)["Seed"].values[0])
