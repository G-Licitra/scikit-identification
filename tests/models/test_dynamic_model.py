# from skmpc.models import DynamicModel
# import matplotlib
# import numpy as np
# import pandas as pd
# import pytest
# import seaborn as sns
# from neuropy.correlation import correlation_analysis
# @pytest.fixture()
# def df_random_small():
#     np.random.seed(2)
#     data = pd.DataFrame(data=np.random.rand(15, 3), columns=["a", "b", "c"])
#     return data
# @pytest.fixture()
# def df_random_large():
#     np.random.seed(150)
#     data = pd.DataFrame(data=np.random.rand(2000, 3), columns=["a", "b", "c"])
#     return data
# @pytest.fixture()
# def df_steady():
#     data = pd.DataFrame(
#         {
#             "a": np.arange(0, 30, 1),
#             "b": np.arange(30, 0, -1),
#             "c": np.arange(0, 15, 0.5),
#         }
#     )
#     return data
# @pytest.fixture()
# def df_with_missings():
#     data = pd.DataFrame(
#         {
#             "a": np.arange(0, 30, 1),
#             "b": np.arange(30, 0, -1),
#             "c": np.arange(0, 15, 0.5),
#             "d": np.arange(0, 60, 2),
#         }
#     )
#     data.iloc[0, 2] = np.nan
#     data.iloc[1, 3] = np.nan
#     return data
# @pytest.fixture()
# def df_with_nonnormal():
#     data = sns.load_dataset("iris")
#     return data
# @pytest.fixture()
# def df_permutation():
#     """Data from http://biol09.biol.umontreal.ca/PLcourses/Statistical_tests.pdf"""
#     data = pd.DataFrame(
#         {
#             "x1": [-2.31, 1.06, 0.76, 1.38, -0.26, 1.29, -1.31, 0.41, -0.67, -0.58],
#             "x2": [-1.08, 1.03, 0.90, 0.24, -0.24, 0.76, -0.57, -0.05, -1.28, 1.04],
#         }
#     )
#     return data
# class TestCorrelationAnalysis:
#     """Test class for function correlation_analysis."""
#     def test_original_df_not_changed(self, df_steady):
#         """Test whether the dataframe stays the same after applying the function"""
#         df_prior = df_steady.copy()
#         mydict, myfig = correlation_analysis(
#             df_steady, permutation_test=False, plot_permutation=False
#         )
#         assert df_prior.equals(df_steady)
#     def test_returned_types(self, df_random_small):
#         """Test whether correct object types are returned"""
#         mydict, myfig = correlation_analysis(
#             df_random_small, permutation_test=False, plot_permutation=False
#         )
#         assert isinstance(mydict, dict)
#         assert not myfig
#         mydict, myfig = correlation_analysis(
#             df_random_small, permutation_test=True, plot_permutation=True
#         )
#         assert isinstance(mydict, dict)
#         assert isinstance(myfig, matplotlib.figure.Figure)
#         assert len(mydict) == 5
#         for key, value in mydict.items():
#             assert key in ["info", "r-value", "p-value", "N", "summary"]
#             assert isinstance(value, pd.DataFrame)
#         assert "permutation-p-value" in mydict["summary"].columns
#         matplotlib.pyplot.close("all")
#     def test_summary_matches_other_dfs(self, df_random_small):
#         """Test whether the summary has the same results as the separate dataframes"""
#         results, _ = correlation_analysis(
#             df_random_small, permutation_test=False, plot_permutation=False
#         )
#         for _, row in results["summary"].iterrows():
#             possible_locations_r = [
#                 results["r-value"].loc[row.feature1, row.feature2],
#                 results["r-value"].loc[row.feature2, row.feature1],
#             ]
#             possible_locations_p = [
#                 results["p-value"].loc[row.feature1, row.feature2],
#                 results["p-value"].loc[row.feature2, row.feature1],
#             ]
#             possible_locations_n = [
#                 results["N"].loc[row.feature1, row.feature2],
#                 results["N"].loc[row.feature2, row.feature1],
#             ]
#             assert row["r-value"] in possible_locations_r
#             assert row["p-value"] in possible_locations_p
#             assert row["N"] in possible_locations_n
#     def test_random_no_correlation(self, df_random_large):
#         """Test whether a random selection of number returns very low correlation numbers"""
#         results, _ = correlation_analysis(
#             df_random_large, permutation_test=True, plot_permutation=False
#         )
#         assert max(abs(results["summary"]["r-value"])) < 0.05
#         assert all([not sign for sign in results["summary"]["stat-sign"]])
#         assert all(
#             [
#                 permutation_p_value > 0.05
#                 for permutation_p_value in results["summary"]["permutation-p-value"]
#             ]
#         )
#     def test_steady_correlation(self, df_steady):
#         """Test whether increasing and decreasing series get coefficient 1 or -1"""
#         # pearson
#         results, _ = correlation_analysis(
#             df_steady, permutation_test=False, plot_permutation=False, method="pearson"
#         )
#         assert all(results["summary"]["stat-sign"])
#         assert all([p < 0.001 for p in results["summary"]["p-value"]])
#         assert all(
#             [abs(r) == pytest.approx(1.0) for r in results["summary"]["r-value"]]
#         )
#         # spearman
#         results, _ = correlation_analysis(
#             df_steady, permutation_test=False, plot_permutation=False, method="spearman"
#         )
#         assert all(results["summary"]["stat-sign"])
#         assert all([p < 0.001 for p in results["summary"]["p-value"]])
#         assert all(
#             [abs(r) == pytest.approx(1.0) for r in results["summary"]["r-value"]]
#         )
#         # kendall
#         results, _ = correlation_analysis(
#             df_steady, permutation_test=False, plot_permutation=False, method="kendall"
#         )
#         assert all(results["summary"]["stat-sign"])
#         assert all([p < 0.001 for p in results["summary"]["p-value"]])
#         assert all(
#             [abs(r) == pytest.approx(1.0) for r in results["summary"]["r-value"]]
#         )
#     def test_missings_pairwise(self, df_with_missings):
#         """Test whether missingness is handled correctly (with setting pairwise)"""
#         number_of_notnans = df_with_missings.notna().sum()
#         results, _ = correlation_analysis(
#             df_with_missings,
#             permutation_test=False,
#             plot_permutation=False,
#             dropna="pairwise",
#         )
#         summary = results["summary"]
#         # summary has valid numbers
#         assert summary.notna().all(axis=None)
#         # 'a' does not have missings, so N is equal to the number of valid values in feature 2
#         assert (
#             summary.loc[summary.feature1 == "a"]
#             .set_index("feature2")["N"]
#             .equals(number_of_notnans[["b", "c", "d"]])
#         )
#         # 'b' does not have missings, so N is equal to the number of valid values in feature 2
#         assert (
#             summary.loc[summary.feature1 == "b"]
#             .set_index("feature2")["N"]
#             .equals(number_of_notnans[["c", "d"]])
#         )
#         # 'c' has one missing, which is on a different place than the missing in 'd'
#         assert (
#             summary.loc[summary.feature1 == "c"]
#             .set_index("feature2")["N"]
#             .equals(number_of_notnans[["d"]] - 1)
#         )
#         # check results
#         assert all(summary["stat-sign"])
#         assert all([p < 0.001 for p in summary["p-value"]])
#         assert all([abs(r) == pytest.approx(1.0) for r in summary["r-value"]])
#     def test_missings_listwise(self, df_with_missings):
#         """Test whether missingness is handled correctly (with setting listwise)"""
#         number_of_notnans = df_with_missings.notna().all(axis=1).sum()
#         results, _ = correlation_analysis(
#             df_with_missings,
#             permutation_test=False,
#             plot_permutation=False,
#             dropna="listwise",
#         )
#         summary = results["summary"]
#         # summary has valid numbers
#         assert summary.notna().all(axis=None)
#         # same N in all correlations
#         assert all(summary["N"] == number_of_notnans)
#         # check results
#         assert all(summary["stat-sign"])
#         assert all([p < 0.001 for p in summary["p-value"]])
#         assert all([abs(r) == pytest.approx(1.0) for r in summary["r-value"]])
#     def test_row_col_list(self, df_with_nonnormal):
#         """Test whether correct features are selected when passing row_list and col_list"""
#         results, _ = correlation_analysis(
#             df_with_nonnormal,
#             permutation_test=False,
#             plot_permutation=False,
#             col_list=["petal_width"],
#             row_list=["sepal_length", "sepal_width"],
#         )
#         assert results["summary"]["feature1"].values.tolist() == [
#             "petal_width",
#             "petal_width",
#         ]
#         assert results["summary"]["feature2"].values.tolist() == [
#             "sepal_length",
#             "sepal_width",
#         ]
#     def test_norm_check(self, df_with_nonnormal):
#         """Test whether Pearson and Spearman correlation analyses are selected when passing check_norm"""
#         results, _ = correlation_analysis(
#             df_with_nonnormal,
#             permutation_test=False,
#             plot_permutation=False,
#             check_norm=True,
#         )
#         assert (
#             results["summary"]["analysis"].values.tolist()
#             == ["Pearson"] + ["Spearman Rank"] * 5
#         )
#     def test_random_seed_permutations(self, df_random_small):
#         """Test whether passing a seed for the randomization of permutations makes sure it is replicable"""
#         result_1, _ = correlation_analysis(
#             df_random_small,
#             permutation_test=True,
#             plot_permutation=False,
#             random_state=25,
#         )
#         result_2, _ = correlation_analysis(
#             df_random_small,
#             permutation_test=True,
#             plot_permutation=False,
#             random_state=25,
#         )
#         # same seed, same results
#         assert result_1["summary"].equals(result_2["summary"])
#         result_3, _ = correlation_analysis(
#             df_random_small,
#             permutation_test=True,
#             plot_permutation=False,
#             random_state=2505,
#         )
#         # different seed, different results
#         assert not result_1["summary"].equals(result_3["summary"])
#     def test_correlation_with_permutation(self, df_permutation):
#         """Test whether correlation values and permutation values are correct"""
#         # check with http://biol09.biol.umontreal.ca/PLcourses/Statistical_tests.pdf
#         results, _ = correlation_analysis(
#             df_permutation,
#             permutation_test=True,
#             plot_permutation=False,
#             random_state=4,
#             n_permutations=1000,
#         )
#         assert results["summary"]["r-value"].iloc[0] == pytest.approx(0.70156, abs=1e-5)
#         assert results["summary"]["p-value"].iloc[0] == pytest.approx(0.0238, abs=1e-4)
#         assert results["summary"]["permutation-p-value"].iloc[0] == pytest.approx(
#             0.026, abs=1e-3
#         )
