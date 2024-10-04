"""Toutes les fonction / traitement back (couche métier)."""
from datalayer.initdata import DataLayer

# Créer une instance de la classe DataLayer
data_layer = DataLayer()

# Charger toutes les données
data_layer.load_data()

# Accéder aux données chargées
interactions_test_df = data_layer.get_interactions_test()
pp_recipes_df = data_layer.get_pp_recipes()
pickle = data_layer.get_ingr_map()

# Afficher quelques exemples de données
print("Exemples d'interactions de test:")
print(interactions_test_df.head())

print("\nExemples de recettes:")
print(pp_recipes_df.head())

print("\nExemples pickles:")
print(pickle.head())
