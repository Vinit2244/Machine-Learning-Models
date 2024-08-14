from "../..models/linear-regression/linear-regression.py" import LinearRegression
# import matplotlib.pyplot as plt

# HEADER = 0 # Index of the header row

# ==================== Reading data in a 2D array ====================
# spotify_csv_path = "../../data/external/spotify.csv"
# spotify_data = []
# with open(spotify_csv_path, "r") as f:
#     for line in f:
#         data_row = line.strip().split(",")
#         new_data_row = list()
#         for col in data_row:
#             new_data_row.append(col.strip())
#         spotify_data.append(new_data_row)

# print("Read Data")

# ==================== Replacing $ with s in data ====================
'''
    The $ character present in the names of the songs was 
    creating error for matplotlib to plot the graphs so 
    replacing it with s in the data.
'''
# for row_idx, row in enumerate(spotify_data):
#     for col_idx, col in enumerate(row):
#         while "$" in col:
#             new_col = ""
#             for character in col:
#                 if character == "$":
#                     new_col += "s"
#                 else:
#                     new_col += character
#             col = new_col
#             spotify_data[row_idx][col_idx] = new_col

# print("Replaced $ with s in data")
# ========================= Helper Functions =========================
# find_col_idx = lambda col_name: spotify_data[HEADER].index(col_name)

# ================== Analysing Distribution of Data ==================
'''Artsits vs No. of Songs is giving issues as there are a large number of artists'''
# # Artist vs No. of Songs
# artists_col_idx = find_col_idx("artists")
# artist_songs_count = dict()
# for row in spotify_data[HEADER+1:]:
#     artists = row[artists_col_idx].split(";")
#     for artist in artists:
#         # Check if artist is already in the dictionary
#         if artist in artist_songs_count:
#             artist_songs_count[artist] += 1
#         else:
#             artist_songs_count[artist] = 1

# x_axis = list(artist_songs_count.keys())    # Names of artists
# y_axis = list(artist_songs_count.values())  # No. of songs by each artist

# plt.bar(x_axis, y_axis)
# plt.xlabel("Artist")
# plt.xticks(rotation=90) # For better readability (ChatGPT prompt: "How to write the values in the axes in matplotlib in vertical fashion")
# plt.ylabel("No. of Songs")
# plt.title("Artist vs No. of Songs")
# artist_vs_n_songs_path = "./figures/artist_vs_n_songs.png"
# # plt.savefig(artist_vs_n_songs_path, format='png')
# plt.show()
# # print("Saved Artist vs No. of Songs figure at", artist_vs_n_songs_path)

# Genre vs No. of Songs
'''There are still a lot of genres - each with 1000 songs - so the plot is not very clear'''
# genre_col_idx = -1
# genre_songs_count = dict()
# for row in spotify_data[HEADER+1:]:
#     genre = row[genre_col_idx]
#     # Check if genre is already in the dictionary
#     if genre in genre_songs_count:
#         genre_songs_count[genre] += 1
#     else:
#         genre_songs_count[genre] = 1

# x_axis = list(genre_songs_count.keys())    # Names of genres
# y_axis = list(genre_songs_count.values())  # No. of songs in each genre

# plt.xlabel("Genre")
# plt.xticks(rotation=90) # For better readability (ChatGPT prompt: "How to write the values in the axes in matplotlib in vertical fashion")
# plt.ylabel("No. of Songs")
# plt.title("Genre vs No. of Songs")
# plt.scatter(x_axis, y_axis, color='red')
# genre_vs_n_songs_path = "./figures/genre_vs_n_songs.svg"
# plt.savefig(genre_vs_n_songs_path, format='svg')
# print("Saved Genre vs No. of Songs figure at", genre_vs_n_songs_path)

# ================== Visualising number of explicit songs ==================
'''Almost all songs are non-explicit, so the pie plot is showing 0% for explicit songs'''
# explicit_col_idx = find_col_idx("explicit")
# n_explicit_songs = 0
# n_non_explicit_songs = 0
# for row in spotify_data[HEADER+1:]:
#     explicit = row[explicit_col_idx]
#     if explicit == "TRUE":
#         n_explicit_songs += 1
#     else:
#         n_non_explicit_songs += 1

# y_vals = [n_explicit_songs, n_non_explicit_songs]
# plt.pie(y_vals, labels=["Explicit Songs", "Non-Explicit Songs"], autopct='%1.1f%%')
# plt.title("Distribution of Explicit and Non-Explicit Songs")
# explicit_vs_non_explicit_path = "./figures/explicit_vs_non_explicit.png"
# plt.savefig(explicit_vs_non_explicit_path, format='png')
# print("Saved Explicit vs Non-Explicit Songs figure at", explicit_vs_non_explicit_path)
# plt.show()

# if __name__ == "__main__":
    