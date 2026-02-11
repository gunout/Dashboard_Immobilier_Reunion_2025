# dashboard_reunion_2025.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import io
import gzip
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Immobilier Réunion 2025",
    page_icon="🏝️",
    layout="wide"
)

# --- Dictionnaire des communes de La Réunion (Code INSEE -> Nom) ---
COMMUNES_REUNION = {
    "97401": "Les Avirons",
    "97402": "Bras Panon",
    "97403": "Cilaos",
    "97404": "Entre-Deux",
    "97405": "L'Étang-Salé",
    "97406": "Petite-Île",
    "97407": "La Plaine-des-Palmistes",
    "97408": "Le Port",
    "97409": "La Possession",
    "97410": "Saint-André",
    "97411": "Saint-Benoît",
    "97412": "Saint-Denis",
    "97413": "Saint-Joseph",
    "97414": "Saint-Leu",
    "97415": "Saint-Louis",
    "97416": "Sainte-Marie",
    "97417": "Sainte-Rose",
    "97418": "Sainte-Suzanne",
    "97419": "Saint-Paul",
    "97420": "Saint-Philippe",
    "97421": "Saint-Pierre",
    "97422": "Salazie",
    "97423": "Le Tampon",
    "97424": "Trois-Bassins",
}

# Inverser le dictionnaire pour avoir Nom -> Code INSEE
NOMS_COMMUNES = {v: k for k, v in COMMUNES_REUNION.items()}

# --- Fonction de chargement des données 2025 pour La Réunion ---
@st.cache_data(ttl=3600)
def load_reunion_2025_data():
    """
    Charge les données DVF 2025 pour toutes les communes de La Réunion
    depuis le fichier départemental compressé
    """
    url = "https://files.data.gouv.fr/geo-dvf/latest/csv/2025/departements/974.csv.gz"
    
    try:
        # Téléchargement du fichier compressé
        with st.spinner("📥 Téléchargement des données DVF 2025 pour La Réunion..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
        
        # Décompression et lecture
        with st.spinner("🔄 Traitement des données..."):
            with gzip.open(io.BytesIO(response.content), 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, sep=',', low_memory=False)
        
        if df.empty:
            st.warning("Aucune donnée trouvée pour La Réunion en 2025")
            return pd.DataFrame()
        
        st.sidebar.success(f"✅ {len(df):,} transactions brutes chargées")
        return df
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            st.error("🚫 Les données 2025 ne sont pas encore disponibles pour La Réunion")
            st.info("📅 Les données DVF sont généralement publiées avec 2-3 mois de décalage")
        else:
            st.error(f"Erreur HTTP : {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return pd.DataFrame()

# --- Fonction de nettoyage et préparation ---
def prepare_data(df):
    """
    Nettoie et prépare les données pour l'analyse
    """
    if df.empty:
        return pd.DataFrame()
    
    # Copie pour éviter les warnings
    df_clean = df.copy()
    
    # Colonnes essentielles
    required_cols = ['date_mutation', 'valeur_fonciere', 'surface_reelle_bati', 
                     'code_postal', 'type_local', 'code_commune']
    
    # Vérification des colonnes disponibles
    available_cols = [col for col in required_cols if col in df_clean.columns]
    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    
    if missing_cols:
        st.sidebar.warning(f"Colonnes manquantes : {', '.join(missing_cols)}")
    
    if 'date_mutation' in df_clean.columns:
        df_clean["date_mutation"] = pd.to_datetime(df_clean["date_mutation"], 
                                                   format='%Y-%m-%d', 
                                                   errors='coerce')
    
    # Conversion des valeurs numériques
    if 'valeur_fonciere' in df_clean.columns:
        df_clean["valeur_fonciere"] = pd.to_numeric(df_clean["valeur_fonciere"], 
                                                    errors='coerce')
    
    if 'surface_reelle_bati' in df_clean.columns:
        df_clean["surface_reelle_bati"] = pd.to_numeric(df_clean["surface_reelle_bati"], 
                                                       errors='coerce')
    
    # Filtrage sur les types de biens
    if 'type_local' in df_clean.columns:
        df_clean = df_clean[df_clean["type_local"].isin(['Maison', 'Appartement'])]
    
    # Suppression des lignes avec valeurs manquantes critiques
    critical_cols = [col for col in ['valeur_fonciere', 'surface_reelle_bati'] 
                    if col in df_clean.columns]
    
    if critical_cols:
        df_clean = df_clean.dropna(subset=critical_cols)
    
    # Filtrage des valeurs aberrantes
    if 'valeur_fonciere' in df_clean.columns:
        df_clean = df_clean[df_clean['valeur_fonciere'] > 10000]  # Min 10k€
        df_clean = df_clean[df_clean['valeur_fonciere'] < 2000000]  # Max 2M€
    
    if 'surface_reelle_bati' in df_clean.columns:
        df_clean = df_clean[df_clean['surface_reelle_bati'] > 9]  # Min 9m²
        df_clean = df_clean[df_clean['surface_reelle_bati'] < 500]  # Max 500m²
    
    # Calcul du prix au m²
    if 'valeur_fonciere' in df_clean.columns and 'surface_reelle_bati' in df_clean.columns:
        df_clean['prix_m2'] = df_clean['valeur_fonciere'] / df_clean['surface_reelle_bati']
        # Filtrage des prix aberrants pour La Réunion
        df_clean = df_clean[(df_clean['prix_m2'] > 500) & (df_clean['prix_m2'] < 8000)]
    
    # Ajout du nom de commune
    if 'code_commune' in df_clean.columns:
        df_clean['code_commune'] = df_clean['code_commune'].astype(str).str.zfill(5)
        df_clean['nom_commune'] = df_clean['code_commune'].map(COMMUNES_REUNION)
        df_clean = df_clean.dropna(subset=['nom_commune'])
    
    return df_clean

# --- Interface Utilisateur ---
st.title("🏝️ Dashboard Immobilier La Réunion - Données 2025")
st.markdown("*Source : data.gouv.fr / DVF*")

# Chargement des données
df_brut = load_reunion_2025_data()

if df_brut.empty:
    st.info("💡 Essayez avec les données 2024 en attendant la mise à jour 2025")
    if st.button("📊 Voir les données 2024"):
        st.switch_page("dashboard_reunion_2024.py")  # À créer si nécessaire
    st.stop()

# Préparation des données
with st.spinner("🧹 Nettoyage et préparation des données..."):
    df = prepare_data(df_brut)

if df.empty:
    st.warning("⚠️ Aucune transaction valide après nettoyage des données")
    
    # Affichage debug optionnel
    if st.checkbox("Afficher les colonnes disponibles"):
        st.write("Colonnes dans le fichier source :")
        st.write(df_brut.columns.tolist())
    st.stop()

st.sidebar.success(f"🏠 {len(df):,} transactions valides")

# --- Sélection de la commune ---
st.sidebar.header("📍 Sélection de la commune")
communes_disponibles = sorted(df['nom_commune'].unique())

if not communes_disponibles:
    st.error("Aucune commune trouvée dans les données")
    st.stop()

selected_commune_name = st.sidebar.selectbox(
    "Choisissez une commune :",
    options=communes_disponibles,
    index=communes_disponibles.index("Saint-Denis") if "Saint-Denis" in communes_disponibles else 0
)

selected_insee_code = NOMS_COMMUNES[selected_commune_name]

# Filtrage par commune
df_commune = df[df['nom_commune'] == selected_commune_name].copy()

if df_commune.empty:
    st.warning(f"Aucune donnée pour {selected_commune_name} en 2025")
    st.stop()

# --- Filtres avancés ---
st.sidebar.header("🔧 Filtres")

# Filtre code postal
if 'code_postal' in df_commune.columns:
    codes_postaux_disponibles = sorted(df_commune['code_postal'].astype(str).unique())
    code_postal_selectionne = st.sidebar.multiselect(
        "Code postal", 
        codes_postaux_disponibles, 
        default=codes_postaux_disponibles
    )
else:
    code_postal_selectionne = []
    st.sidebar.warning("Code postal non disponible")

# Filtre type de bien
if 'type_local' in df_commune.columns:
    type_local_options = ['Tous', 'Maison', 'Appartement']
    type_local = st.sidebar.selectbox("Type de bien", type_local_options)
else:
    type_local = 'Tous'

# Filtre prix
prix_min = st.sidebar.number_input(
    "Prix minimum (€)", 
    value=0, 
    step=10000,
    min_value=0
)
prix_max = st.sidebar.number_input(
    "Prix maximum (€)", 
    value=int(df_commune['valeur_fonciere'].max()), 
    step=10000,
    min_value=0
)

# Application des filtres
df_filtre = df_commune.copy()

if code_postal_selectionne and 'code_postal' in df_filtre.columns:
    df_filtre = df_filtre[df_filtre['code_postal'].astype(str).isin(code_postal_selectionne)]

df_filtre = df_filtre[
    (df_filtre['valeur_fonciere'] >= prix_min) & 
    (df_filtre['valeur_fonciere'] <= prix_max)
]

if type_local != 'Tous' and 'type_local' in df_filtre.columns:
    df_filtre = df_filtre[df_filtre['type_local'] == type_local]

if df_filtre.empty:
    st.warning("Aucune transaction ne correspond à vos filtres.")
    st.stop()

# --- KPIs ---
st.header(f"📊 Indicateurs Clés - {selected_commune_name}")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    prix_m2_moyen = df_filtre['prix_m2'].mean()
    st.metric(
        "Prix moyen / m²", 
        f"{prix_m2_moyen:,.0f} €",
        delta=None
    )

with col2:
    prix_median = df_filtre['valeur_fonciere'].median()
    st.metric(
        "Prix médian", 
        f"{prix_median:,.0f} €"
    )

with col3:
    nb_transactions = len(df_filtre)
    st.metric(
        "Transactions", 
        f"{nb_transactions:,}"
    )

with col4:
    surface_moyenne = df_filtre['surface_reelle_bati'].mean()
    st.metric(
        "Surface moyenne", 
        f"{surface_moyenne:.0f} m²"
    )

with col5:
    if 'nombre_pieces_principales' in df_filtre.columns:
        pieces_moyennes = df_filtre['nombre_pieces_principales'].mean()
        st.metric(
            "Pièces principales", 
            f"{pieces_moyennes:.1f}"
        )

# --- Visualisations ---
st.header(f"📈 Analyses - {selected_commune_name}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution des prix au m²")
    if not df_filtre.empty and 'prix_m2' in df_filtre.columns:
        fig = px.histogram(
            df_filtre, 
            x='prix_m2', 
            nbins=30,
            color='type_local' if 'type_local' in df_filtre.columns else None,
            marginal="box",
            title=f"Prix au m² - {selected_commune_name}",
            labels={'prix_m2': 'Prix au m² (€)', 'count': 'Nombre de transactions'}
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Répartition par type de bien")
    if 'type_local' in df_filtre.columns:
        type_counts = df_filtre['type_local'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Nombre']
        fig = px.pie(
            type_counts, 
            values='Nombre', 
            names='Type',
            title=f"Répartition Maison/Appartement - {selected_commune_name}",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Carte ---
st.subheader(f"🗺️ Carte des transactions - {selected_commune_name}")

if 'latitude' in df_filtre.columns and 'longitude' in df_filtre.columns:
    df_carte = df_filtre.dropna(subset=['latitude', 'longitude'])
    
    if not df_carte.empty:
        # Limiter à 1000 points pour la performance
        if len(df_carte) > 1000:
            df_carte = df_carte.sample(1000)
            st.caption(f"Affichage de 1000 transactions sur {len(df_filtre)} (échantillon aléatoire)")
        
        fig = px.scatter_mapbox(
            df_carte,
            lat="latitude",
            lon="longitude",
            color="prix_m2",
            size="surface_reelle_bati",
            hover_data={
                "valeur_fonciere": ":.0f",
                "type_local": True,
                "date_mutation": True,
                "surface_reelle_bati": ":.0f",
                "prix_m2": ":.0f"
            },
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=11,
            mapbox_style="open-street-map",
            title=f"Transactions à {selected_commune_name}",
            labels={"prix_m2": "Prix/m² (€)"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📍 Données de géolocalisation non disponibles pour ces transactions")
else:
    st.info("🗺️ Les données de localisation ne sont pas disponibles dans ce fichier")

# --- Évolution temporelle ---
st.subheader(f"📅 Évolution des prix et volumes - {selected_commune_name}")

if 'date_mutation' in df_filtre.columns and not df_filtre.empty:
    # Agrégation mensuelle
    df_filtre['mois'] = df_filtre['date_mutation'].dt.to_period('M')
    df_mensuel = df_filtre.groupby('mois').agg({
        'prix_m2': 'mean',
        'valeur_fonciere': ['count', 'mean']
    }).round(0)
    
    df_mensuel.columns = ['prix_m2_moyen', 'nb_transactions', 'prix_moyen']
    df_mensuel = df_mensuel.reset_index()
    df_mensuel['mois'] = df_mensuel['mois'].astype(str)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            df_mensuel,
            x='mois',
            y='prix_m2_moyen',
            title="Évolution du prix au m² moyen",
            markers=True,
            labels={'mois': 'Mois', 'prix_m2_moyen': 'Prix moyen au m² (€)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_mensuel,
            x='mois',
            y='nb_transactions',
            title="Nombre de transactions par mois",
            labels={'mois': 'Mois', 'nb_transactions': 'Nombre de transactions'}
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Tableau des transactions ---
st.subheader("📋 Détail des dernières transactions")

# Colonnes à afficher
display_cols = ['date_mutation', 'valeur_fonciere', 'surface_reelle_bati', 
                'prix_m2', 'type_local', 'code_postal', 'nombre_pieces_principales']

available_display_cols = [col for col in display_cols if col in df_filtre.columns]

if available_display_cols:
    df_display = df_filtre.sort_values('date_mutation', ascending=False).head(100)
    
    # Formatage des colonnes
    if 'valeur_fonciere' in df_display.columns:
        df_display['valeur_fonciere'] = df_display['valeur_fonciere'].apply(
            lambda x: f"{x:,.0f} €"
        )
    if 'prix_m2' in df_display.columns:
        df_display['prix_m2'] = df_display['prix_m2'].apply(
            lambda x: f"{x:,.0f} €/m²"
        )
    if 'surface_reelle_bati' in df_display.columns:
        df_display['surface_reelle_bati'] = df_display['surface_reelle_bati'].apply(
            lambda x: f"{x:.0f} m²"
        )
    
    st.dataframe(
        df_display[available_display_cols],
        use_container_width=True,
        hide_index=True
    )
    
    st.caption(f"Affichage des 100 dernières transactions sur {len(df_filtre)}")

# --- Pied de page ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; padding: 10px;'>
        <b>Source des données :</b> data.gouv.fr - Demande de Valeurs Foncières (DVF) 2025<br>
        <b>Mise à jour :</b> Fichier départemental La Réunion (974)<br>
        <b>Note :</b> Les données 2025 sont mises à jour progressivement par la DGFiP
    </div>
    """,
    unsafe_allow_html=True
)
