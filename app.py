import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="AI4I - Maintenance prédictive",
    page_icon="",
    layout="wide"
)

st.title(" maintenance prédictive")

uploaded_file = st.file_uploader(
    "Importer le dataset AI4I au format CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Importe le fichier CSV pour commencer l'exploration.")
else:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Impossible de lire le fichier CSV : {e}")
        st.stop()


    st.header("Module 1 — Exploration des données")

    st.subheader("1. Aperçu des données")
    st.dataframe(df.head(10), width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de lignes", df.shape[0])
    col2.metric("Nombre de colonnes", df.shape[1])
    col3.metric("Valeurs manquantes", int(df.isnull().sum().sum()))

    st.subheader("2. Statistiques descriptives")
    st.dataframe(df.describe(include="all").T, width="stretch")

    if "Machine failure" in df.columns:
        st.subheader("3. Répartition des défaillances")
        counts = (
            df["Machine failure"]
            .map({0: "Aucune défaillance", 1: "Défaillance"})
            .fillna(df["Machine failure"].astype(str))
            .value_counts()
            .rename_axis("État")
            .reset_index(name="Nombre")
        )
        fig = px.pie(
            counts, names="État", values="Nombre",
            title="Répartition des machines"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("4. Distribution d'une variable")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    excluded_for_exploration = {
        "UDI", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"
    }
    numeric_features = [
        c for c in numeric_columns if c not in excluded_for_exploration
    ]

    if numeric_features:
        variable = st.selectbox(
            "Choisir une variable numérique",
            numeric_features,
            key="exploration_numeric_variable"
        )
        fig = px.histogram(
            df, x=variable, nbins=30,
            title=f"Distribution de {variable}"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("5. Matrice de corrélation")
    corr_columns = [c for c in numeric_features if c in df.columns]
    if len(corr_columns) >= 2:
        corr = df[corr_columns].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="Corrélations entre variables numériques",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Il faut au moins deux variables numériques pour afficher la corrélation.")

    
    
    st.divider()
    st.header("Module 2 — Préparation des données et équilibrage SMOTE")

    st.subheader("1. Choix de la cible et des variables")

    excluded_columns = {
        "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"
    }
    target_default = (
        df.columns.get_loc("Machine failure")
        if "Machine failure" in df.columns
        else 0
    )

    target = st.selectbox(
        "Choisir la variable cible",
        options=list(df.columns),
        index=target_default,
        key="m2_target"
    )

    default_features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    available_features = [
        c for c in df.columns
        if c != target and c not in excluded_columns
    ]
    default_selected_features = [
        c for c in default_features if c in available_features
    ]

    selected_features = st.multiselect(
        "Choisir les variables explicatives",
        options=available_features,
        default=default_selected_features,
        key="m2_features"
    )

    st.subheader("2. Traitement des données")
    missing_method = st.selectbox(
        "Traitement des valeurs manquantes",
        ["Suppression des lignes", "Imputation (médiane / mode)"],
        key="m2_missing"
    )
    scaling_method = st.selectbox(
        "Mise à l'échelle des variables numériques",
        ["Standardisation", "Min-Max", "Aucune"],
        key="m2_scaling"
    )

    st.subheader("3. Séparation entraînement / test")
    col_a, col_b = st.columns(2)
    with col_a:
        test_size_percent = st.slider(
            "Pourcentage réservé au test",
            min_value=10, max_value=40, value=20, step=5,
            key="m2_test_size"
        )
    with col_b:
        random_state = st.number_input(
            "Random state",
            min_value=0, max_value=10000, value=42, step=1,
            key="m2_random_state"
        )

    use_smote = st.checkbox(
        "Appliquer SMOTE aux données d'entraînement",
        value=True,
        key="m2_smote"
    )

    if st.button("Préparer les données", type="primary", key="m2_prepare"):
        if not selected_features:
            st.error("Choisis au moins une variable explicative.")
            st.stop()

        data = df[selected_features + [target]].copy()
        data = data.dropna(subset=[target])

        if data[target].nunique(dropna=True) != 2:
            st.error(
                "Cette version attend une cible de classification binaire "
                "comportant exactement deux classes."
            )
            st.stop()

        if missing_method == "Suppression des lignes":
            data = data.dropna(subset=selected_features)

        if data.empty:
            st.error("Aucune ligne ne reste après le traitement des valeurs manquantes.")
            st.stop()

        X = data[selected_features].copy()
        y_codes, target_labels = pd.factorize(data[target], sort=True)
        y = pd.Series(y_codes, index=data.index, name=target)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size_percent / 100,
                random_state=int(random_state),
                stratify=y
            )
        except ValueError as e:
            st.error(
                "Impossible de réaliser la séparation stratifiée. "
                f"Vérifie le nombre d'observations par classe. Détail : {e}"
            )
            st.stop()

        numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = [
            c for c in X_train.columns if c not in numeric_cols
        ]

        if missing_method == "Imputation (médiane / mode)":
            all_missing = [
                c for c in X_train.columns if X_train[c].isna().all()
            ]
            if all_missing:
                st.error(
                    "Ces variables sont entièrement vides dans le jeu "
                    f"d'entraînement : {', '.join(all_missing)}. "
                    "Retire-les ou choisis une autre stratégie."
                )
                st.stop()

            if numeric_cols:
                num_imputer = SimpleImputer(strategy="median")
                X_train.loc[:, numeric_cols] = num_imputer.fit_transform(
                    X_train[numeric_cols]
                )
                X_test.loc[:, numeric_cols] = num_imputer.transform(
                    X_test[numeric_cols]
                )

            if categorical_cols:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                X_train.loc[:, categorical_cols] = cat_imputer.fit_transform(
                    X_train[categorical_cols]
                )
                X_test.loc[:, categorical_cols] = cat_imputer.transform(
                    X_test[categorical_cols]
                )

 
        numeric_original_cols = numeric_cols.copy()

        X_train = pd.get_dummies(
            X_train, columns=categorical_cols, dtype=int
        )
        X_test = pd.get_dummies(
            X_test, columns=categorical_cols, dtype=int
        )
        X_train, X_test = X_train.align(
            X_test, join="left", axis=1, fill_value=0
        )

        numeric_to_scale = [
            c for c in numeric_original_cols if c in X_train.columns
        ]
        if scaling_method != "Aucune" and numeric_to_scale:
            scaler = (
                StandardScaler()
                if scaling_method == "Standardisation"
                else MinMaxScaler()
            )
            X_train[numeric_to_scale] = X_train[numeric_to_scale].astype(float)
            X_test[numeric_to_scale] = X_test[numeric_to_scale].astype(float)

            X_train.loc[:, numeric_to_scale] = scaler.fit_transform(
                X_train[numeric_to_scale]
            )
            X_test.loc[:, numeric_to_scale] = scaler.transform(
                X_test[numeric_to_scale]
            )

        counts_before = y_train.value_counts().sort_index()

        if use_smote:
            min_class_count = int(counts_before.min())
            if min_class_count < 2:
                st.error(
                    "SMOTE nécessite au moins deux observations "
                    "dans chaque classe d'entraînement."
                )
                st.stop()

            smote = SMOTE(
                random_state=int(random_state),
                k_neighbors=min(5, min_class_count - 1)
            )
            try:
                X_train_resampled, y_train_resampled = smote.fit_resample(
                    X_train, y_train
                )
            except ValueError as e:
                st.error(f"Erreur pendant SMOTE : {e}")
                st.stop()
        else:
            X_train_resampled = X_train.copy()
            y_train_resampled = y_train.copy()

        counts_after = pd.Series(y_train_resampled).value_counts().sort_index()
        class_names = {
            i: str(label) for i, label in enumerate(target_labels)
        }

        before_df = pd.DataFrame({
            "Classe": [
                class_names.get(int(i), str(i)) for i in counts_before.index
            ],
            "Nombre": counts_before.values,
            "Étape": "Avant SMOTE"
        })
        after_df = pd.DataFrame({
            "Classe": [
                class_names.get(int(i), str(i)) for i in counts_after.index
            ],
            "Nombre": counts_after.values,
            "Étape": "Après SMOTE" if use_smote else "Sans SMOTE"
        })
        comparison = pd.concat([before_df, after_df], ignore_index=True)

        st.subheader("4. Répartition des classes")
        fig = px.bar(
            comparison,
            x="Étape",
            y="Nombre",
            color="Classe",
            barmode="group",
            text="Nombre",
            title="Répartition des classes d'entraînement"
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(comparison, hide_index=True, width="stretch")

        st.success("Préparation des données terminée.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Train avant SMOTE", len(y_train))
        col2.metric("Train après SMOTE", len(y_train_resampled))
        col3.metric("Test inchangé", len(y_test))


        st.session_state["X_train_resampled"] = X_train_resampled
        st.session_state["y_train_resampled"] = y_train_resampled
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["target_labels"] = list(target_labels)

        st.subheader("5. Aperçu des variables préparées")
        st.dataframe(
            pd.DataFrame(X_train_resampled).head(10),
            width="stretch"
        )
    
   




import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

st.header("Module 3 : Entraînement et évaluation du modèle XGBoost")




cles_requises = [
    "X_train_resampled",
    "y_train_resampled",
    "X_test",
    "y_test"
]

if not all(cle in st.session_state for cle in cles_requises):

    st.warning(
        "Les données préparées ne sont pas disponibles. "
        "Veuillez d'abord exécuter le Module 2."
    )

else:

    X_train = st.session_state["X_train_resampled"]
    y_train = st.session_state["y_train_resampled"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    
    
    X_train_xgb = pd.DataFrame(X_train).copy()
    X_test_xgb = pd.DataFrame(X_test).copy()

    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    
    if X_train_xgb.shape[1] != X_test_xgb.shape[1]:

        st.error(
            "Les données d'entraînement et de test "
            "n'ont pas le même nombre de variables."
        )

    elif y_train.nunique() != 2 or y_test.nunique() != 2:

        st.error(
            "Ce module est configuré pour une classification binaire. "
            "Vérifiez que la variable cible contient deux classes."
        )

    else:

        
     

        noms_originaux = [
            str(col) for col in X_train_xgb.columns
        ]

        def nettoyer_noms_colonnes(columns):
            noms_nettoyes = []
            compteur = {}

            for col in columns:
                nom = str(col)

                nom = re.sub(r"[\[\]<]", "_", nom)

                nom = re.sub(r"[^A-Za-z0-9_]", "_", nom)

                if not nom:
                    nom = "variable"

                if nom in compteur:
                    compteur[nom] += 1
                    nom = f"{nom}_{compteur[nom]}"
                else:
                    compteur[nom] = 0

                noms_nettoyes.append(nom)

            return noms_nettoyes

        noms_xgb = nettoyer_noms_colonnes(
            X_train_xgb.columns
        )

        X_train_xgb.columns = noms_xgb
        X_test_xgb.columns = noms_xgb

       
       

        st.subheader("3.1. Données d'entraînement et de test")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Lignes d'entraînement",
                X_train_xgb.shape[0]
            )

        with col2:
            st.metric(
                "Lignes de test",
                X_test_xgb.shape[0]
            )

        with col3:
            st.metric(
                "Nombre de variables",
                X_train_xgb.shape[1]
            )


      
       
        st.subheader("3.2. Configuration du modèle")

        col1, col2, col3 = st.columns(3)

        with col1:
            n_estimators = st.slider(
                "Nombre d'arbres (n_estimators)",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                help="Nombre d'arbres construits par le modèle."
            )

        with col2:
            max_depth = st.slider(
                "Profondeur maximale (max_depth)",
                min_value=2,
                max_value=12,
                value=5,
                step=1,
                help="Contrôle la complexité de chaque arbre."
            )

        with col3:
            learning_rate = st.select_slider(
                "Taux d'apprentissage (learning_rate)",
                options=[0.01, 0.03, 0.05, 0.1, 0.2],
                value=0.1,
                help="Contrôle la contribution de chaque arbre."
            )

        st.write(
            "**Hyperparamètres sélectionnés :**",
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            }
        )

       
       

        st.subheader("3.3. Entraînement du modèle")

        if st.button(
            "Entraîner le modèle XGBoost",
            type="primary",
            key="bouton_entrainement_xgboost"
        ):

            with st.spinner(
                "Entraînement du modèle en cours..."
            ):

               
                nombre_negatifs = int((y_train == 0).sum())
                nombre_positifs = int((y_train == 1).sum())

                if nombre_positifs == 0:
                    st.error(
                        "Aucun exemple positif dans les données "
                        "d'entraînement."
                    )
                    st.stop()

                poids_classes = (
                    nombre_negatifs / nombre_positifs
                )

                model = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                    scale_pos_weight=poids_classes
                )

               
                model.fit(X_train_xgb, y_train)

                y_pred = model.predict(X_test_xgb)

                y_proba = model.predict_proba(
                    X_test_xgb
                )[:, 1]

              
               

                accuracy = accuracy_score(y_test, y_pred)

                precision = precision_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0
                )

                try:
                    roc_auc = roc_auc_score(
                        y_test,
                        y_proba
                    )
                except ValueError:
                    roc_auc = np.nan

                try:
                    pr_auc = average_precision_score(
                        y_test,
                        y_proba
                    )
                except ValueError:
                    pr_auc = np.nan

                st.session_state["xgb_model"] = model
                st.session_state["xgb_predictions"] = y_pred
                st.session_state["xgb_probabilities"] = y_proba
                st.session_state["xgb_metrics"] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-score": f1,
                    "ROC-AUC": roc_auc,
                    "PR-AUC": pr_auc
                }

            st.success(
                "Le modèle XGBoost a été entraîné avec succès."
            )

        
     

        if "xgb_model" in st.session_state:

            model = st.session_state["xgb_model"]
            y_pred = st.session_state["xgb_predictions"]
            y_proba = st.session_state["xgb_probabilities"]
            metrics = st.session_state["xgb_metrics"]

            st.subheader("3.4. Résultats de l'évaluation")

            st.write(
                "Les métriques suivantes sont calculées sur "
                "le jeu de test."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{metrics['Accuracy']:.2%}"
                )

            with col2:
                st.metric(
                    "Precision",
                    f"{metrics['Precision']:.2%}"
                )

            with col3:
                st.metric(
                    "Recall",
                    f"{metrics['Recall']:.2%}"
                )

            col4, col5, col6 = st.columns(3)

            with col4:
                st.metric(
                    "F1-score",
                    f"{metrics['F1-score']:.2%}"
                )

            with col5:
                valeur_roc = metrics["ROC-AUC"]
                st.metric(
                    "ROC-AUC",
                    f"{valeur_roc:.2%}"
                    if not np.isnan(valeur_roc)
                    else "Non disponible"
                )

            with col6:
                valeur_pr = metrics["PR-AUC"]
                st.metric(
                    "PR-AUC",
                    f"{valeur_pr:.2%}"
                    if not np.isnan(valeur_pr)
                    else "Non disponible"
                )

            with st.expander(
                "Comprendre les métriques"
            ):
                st.markdown(
                    """
                    - **Accuracy :** proportion totale de prédictions correctes.
                    - **Precision :** parmi les défaillances prédites,
                      proportion de défaillances réellement observées.
                    - **Recall :** proportion des défaillances réelles
                      correctement détectées.
                    - **F1-score :** équilibre entre précision et rappel.
                    - **ROC-AUC :** capacité à distinguer les deux classes
                      sur différents seuils.
                    - **PR-AUC :** mesure particulièrement utile lorsque
                      les défaillances sont rares.
                    """
                )

          
            

            st.subheader("3.5. Matrice de confusion")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=[
                    "Prédit : aucune défaillance",
                    "Prédit : défaillance"
                ],
                y=[
                    "Réel : aucune défaillance",
                    "Réel : défaillance"
                ],
                labels={
                    "x": "Classe prédite",
                    "y": "Classe réelle",
                    "color": "Nombre"
                },
                title="Matrice de confusion - XGBoost",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(
                fig_cm,
                use_container_width=True
            )


            
            

            st.subheader("3.6. Rapport de classification")

            rapport = classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "Aucune défaillance (0)",
                    "Défaillance (1)"
                ],
                output_dict=True,
                zero_division=0
            )

            rapport_df = pd.DataFrame(
                rapport
            ).transpose()

            st.dataframe(
                rapport_df.style.format("{:.3f}"),
                use_container_width=True
            )

           
            

            st.subheader("3.7. Courbe ROC")

            try:
                fpr, tpr, seuils_roc = roc_curve(
                    y_test,
                    y_proba
                )

                fig_roc = go.Figure()

                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"XGBoost (AUC = {metrics['ROC-AUC']:.3f})"
                    )
                )

                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Classification aléatoire",
                        line=dict(dash="dash")
                    )
                )

                fig_roc.update_layout(
                    title="Courbe ROC",
                    xaxis_title="Taux de faux positifs",
                    yaxis_title="Taux de vrais positifs",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1])
                )

                st.plotly_chart(
                    fig_roc,
                    use_container_width=True
                )

            except ValueError:
                st.info(
                    "La courbe ROC n'est pas disponible "
                    "pour ce jeu de test."
                )

           
            

            st.subheader("3.8. Courbe Precision-Recall")

            precision_curve, recall_curve, seuils_pr = (
                precision_recall_curve(
                    y_test,
                    y_proba
                )
            )

            fig_pr = go.Figure()

            fig_pr.add_trace(
                go.Scatter(
                    x=recall_curve,
                    y=precision_curve,
                    mode="lines",
                    name="XGBoost"
                )
            )

            fig_pr.update_layout(
                title="Courbe Precision-Recall",
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )

            st.plotly_chart(
                fig_pr,
                use_container_width=True
            )

            
          

            st.subheader("3.9. Importance des variables")

            importances = model.feature_importances_

            importance_df = pd.DataFrame({
                "Variable": noms_originaux,
                "Importance": importances
            }).sort_values(
                by="Importance",
                ascending=False
            )

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Variable",
                orientation="h",
                title="Importance des variables dans XGBoost",
                color="Importance",
                color_continuous_scale="Blues"
            )

            fig_importance.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

            st.dataframe(
                importance_df.reset_index(drop=True),
                use_container_width=True
            )

          




import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

st.header("Module 4 : Entraînement et évaluation du modèle Random Forest")

st.write(
    """
    Random Forest est un algorithme d'apprentissage supervisé qui combine
    plusieurs arbres de décision pour réaliser une classification.
    Nous l'utilisons ici pour prédire les défaillances des machines.
    """
)




cles_requises_rf = [
    "X_train_resampled",
    "y_train_resampled",
    "X_test",
    "y_test"
]

if not all(cle in st.session_state for cle in cles_requises_rf):

    st.warning(
        "Veuillez d'abord exécuter le Module 2 pour préparer "
        "les données d'entraînement et de test."
    )

else:

    X_train_rf = pd.DataFrame(
        st.session_state["X_train_resampled"]
    ).copy()

    X_test_rf = pd.DataFrame(
        st.session_state["X_test"]
    ).copy()

    y_train_rf = pd.Series(
        st.session_state["y_train_resampled"]
    ).reset_index(drop=True)

    y_test_rf = pd.Series(
        st.session_state["y_test"]
    ).reset_index(drop=True)

    if X_train_rf.shape[1] != X_test_rf.shape[1]:

        st.error(
            "Les données d'entraînement et de test "
            "n'ont pas le même nombre de variables."
        )

    elif y_train_rf.nunique() != 2 or y_test_rf.nunique() != 2:

        st.error(
            "Ce module nécessite une cible de classification binaire."
        )

    else:

        
       

        st.subheader("4.1. Configuration des hyperparamètres")

        col1, col2, col3 = st.columns(3)

        with col1:
            n_estimators_rf = st.slider(
                "Nombre d'arbres",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                key="rf_n_estimators"
            )

        with col2:
            max_depth_rf = st.selectbox(
                "Profondeur maximale",
                options=[None, 5, 10, 15, 20, 30],
                index=3,
                format_func=lambda x: (
                    "Illimitée" if x is None else str(x)
                ),
                key="rf_max_depth"
            )

        with col3:
            min_samples_split_rf = st.select_slider(
                "Échantillons minimum pour diviser un nœud",
                options=[2, 5, 10, 20],
                value=2,
                key="rf_min_samples_split"
            )

        class_weight_rf = st.selectbox(
            "Gestion du déséquilibre des classes",
            options=["Aucune", "Balanced"],
            index=0,
            key="rf_class_weight",
            help=(
                "Choisir Balanced pour pondérer automatiquement "
                "les classes selon leur fréquence."
            )
        )

      
       

        st.subheader("4.2. Entraînement du modèle")

        if st.button(
            "Entraîner le modèle Random Forest",
            type="primary",
            key="bouton_entrainement_random_forest"
        ):

            with st.spinner(
                "Entraînement du modèle Random Forest en cours..."
            ):

                poids_rf = (
                    "balanced"
                    if class_weight_rf == "Balanced"
                    else None
                )

                model_rf = RandomForestClassifier(
                    n_estimators=n_estimators_rf,
                    max_depth=max_depth_rf,
                    min_samples_split=min_samples_split_rf,
                    class_weight=poids_rf,
                    random_state=42,
                    n_jobs=-1
                )

               
                model_rf.fit(X_train_rf, y_train_rf)

                y_pred_rf = model_rf.predict(X_test_rf)

                classes_rf = list(model_rf.classes_)
                index_classe_positive = classes_rf.index(1)

                y_proba_rf = model_rf.predict_proba(
                    X_test_rf
                )[:, index_classe_positive]

                accuracy_rf = accuracy_score(
                    y_test_rf, y_pred_rf
                )

                precision_rf = precision_score(
                    y_test_rf,
                    y_pred_rf,
                    pos_label=1,
                    zero_division=0
                )

                recall_rf = recall_score(
                    y_test_rf,
                    y_pred_rf,
                    pos_label=1,
                    zero_division=0
                )

                f1_rf = f1_score(
                    y_test_rf,
                    y_pred_rf,
                    pos_label=1,
                    zero_division=0
                )

                roc_auc_rf = roc_auc_score(
                    y_test_rf, y_proba_rf
                )

                pr_auc_rf = average_precision_score(
                    y_test_rf, y_proba_rf
                )

                st.session_state["rf_model"] = model_rf
                st.session_state["rf_predictions"] = y_pred_rf
                st.session_state["rf_probabilities"] = y_proba_rf

                st.session_state["rf_metrics"] = {
                    "Accuracy": accuracy_rf,
                    "Precision": precision_rf,
                    "Recall": recall_rf,
                    "F1-score": f1_rf,
                    "ROC-AUC": roc_auc_rf,
                    "PR-AUC": pr_auc_rf
                }

                st.session_state["rf_feature_names"] = [
                    str(col) for col in X_train_rf.columns
                ]

            st.success(
                "Le modèle Random Forest a été entraîné avec succès."
            )

     
   

        if "rf_model" in st.session_state:

            model_rf = st.session_state["rf_model"]
            y_pred_rf = st.session_state["rf_predictions"]
            y_proba_rf = st.session_state["rf_probabilities"]
            metrics_rf = st.session_state["rf_metrics"]

            st.subheader("4.3. Métriques d'évaluation")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{metrics_rf['Accuracy']:.2%}"
                )

            with col2:
                st.metric(
                    "Precision",
                    f"{metrics_rf['Precision']:.2%}"
                )

            with col3:
                st.metric(
                    "Recall",
                    f"{metrics_rf['Recall']:.2%}"
                )

            col4, col5, col6 = st.columns(3)

            with col4:
                st.metric(
                    "F1-score",
                    f"{metrics_rf['F1-score']:.2%}"
                )

            with col5:
                st.metric(
                    "ROC-AUC",
                    f"{metrics_rf['ROC-AUC']:.2%}"
                )

            with col6:
                st.metric(
                    "PR-AUC",
                    f"{metrics_rf['PR-AUC']:.2%}"
                )

            
           

            st.subheader("4.4. Matrice de confusion")

            cm_rf = confusion_matrix(
                y_test_rf,
                y_pred_rf,
                labels=[0, 1]
            )

            fig_cm_rf = px.imshow(
                cm_rf,
                text_auto=True,
                x=["Prédit : 0", "Prédit : 1"],
                y=["Réel : 0", "Réel : 1"],
                labels={
                    "x": "Classe prédite",
                    "y": "Classe réelle",
                    "color": "Nombre"
                },
                title="Matrice de confusion - Random Forest",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(
                fig_cm_rf,
                use_container_width=True
            )

          
   
            st.subheader("4.5. Rapport de classification")

            rapport_rf = classification_report(
                y_test_rf,
                y_pred_rf,
                labels=[0, 1],
                target_names=[
                    "Aucune défaillance (0)",
                    "Défaillance (1)"
                ],
                output_dict=True,
                zero_division=0
            )

            rapport_rf_df = pd.DataFrame(
                rapport_rf
            ).transpose()

            st.dataframe(
                rapport_rf_df.style.format("{:.3f}"),
                use_container_width=True
            )

         
            

            st.subheader("4.6. Courbe ROC")

            fpr_rf, tpr_rf, _ = roc_curve(
                y_test_rf,
                y_proba_rf
            )

            fig_roc_rf = go.Figure()

            fig_roc_rf.add_trace(
                go.Scatter(
                    x=fpr_rf,
                    y=tpr_rf,
                    mode="lines",
                    name=f"Random Forest (AUC = {metrics_rf['ROC-AUC']:.3f})"
                )
            )

            fig_roc_rf.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Classification aléatoire",
                    line=dict(dash="dash")
                )
            )

            fig_roc_rf.update_layout(
                title="Courbe ROC - Random Forest",
                xaxis_title="Taux de faux positifs",
                yaxis_title="Taux de vrais positifs",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )

            st.plotly_chart(
                fig_roc_rf,
                use_container_width=True
            )

        
            

            st.subheader("4.7. Courbe Precision-Recall")

            precision_rf_curve, recall_rf_curve, _ = (
                precision_recall_curve(
                    y_test_rf,
                    y_proba_rf
                )
            )

            fig_pr_rf = go.Figure()

            fig_pr_rf.add_trace(
                go.Scatter(
                    x=recall_rf_curve,
                    y=precision_rf_curve,
                    mode="lines",
                    name="Random Forest"
                )
            )

            fig_pr_rf.update_layout(
                title="Courbe Precision-Recall - Random Forest",
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )

            st.plotly_chart(
                fig_pr_rf,
                use_container_width=True
            )

       
        

            st.subheader("4.8. Importance des variables")

            noms_variables_rf = st.session_state[
                "rf_feature_names"
            ]

            importance_rf_df = pd.DataFrame({
                "Variable": noms_variables_rf,
                "Importance": model_rf.feature_importances_
            }).sort_values(
                by="Importance",
                ascending=False
            )

            fig_importance_rf = px.bar(
                importance_rf_df,
                x="Importance",
                y="Variable",
                orientation="h",
                title="Importance des variables - Random Forest",
                color="Importance",
                color_continuous_scale="Blues"
            )

            fig_importance_rf.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig_importance_rf,
                use_container_width=True
            )

            st.dataframe(
                importance_rf_df.reset_index(drop=True),
                use_container_width=True
            )


   


import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


st.header("Module 5 : Autres modèles de classification")

st.write(
    """
    Entraînement de trois modèles supplémentaires :
    **Régression logistique, Arbre de décision et KNN**.
    Ils seront évalués sur le même jeu de test que Random Forest
    et XGBoost.
    """
)

cles_requises = [
    "X_train_resampled",
    "y_train_resampled",
    "X_test",
    "y_test"
]

if not all(k in st.session_state for k in cles_requises):

    st.warning(
        "Veuillez d'abord exécuter le Module 2 pour préparer les données."
    )

else:

    X_train = pd.DataFrame(
        st.session_state["X_train_resampled"]
    ).copy()

    X_test = pd.DataFrame(
        st.session_state["X_test"]
    ).copy()

    y_train = pd.Series(
        st.session_state["y_train_resampled"]
    ).reset_index(drop=True)

    y_test = pd.Series(
        st.session_state["y_test"]
    ).reset_index(drop=True)

    if not X_train.columns.equals(X_test.columns):
        st.error(
            "Les variables d'entraînement et de test ne correspondent pas. "
            "Vérifiez la préparation des données dans le Module 2."
        )
        st.stop()

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        st.error(
            "Ce code est conçu pour une classification binaire "
            "avec les classes 0 et 1."
        )
        st.stop()

    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        st.error(
            "La cible doit être encodée en 0 et 1, "
            "avec 1 représentant la panne."
        )
        st.stop()


    st.subheader("5.1. Régression logistique")

    col1, col2 = st.columns(2)

    with col1:
        lr_C = st.select_slider(
            "Régularisation C",
            options=[0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0,
            key="lr_C"
        )

    with col2:
        lr_weight_label = st.selectbox(
            "Pondération des classes - Logistic Regression",
            options=["Aucune", "Balanced"],
            key="lr_class_weight"
        )


    st.subheader("5.2. Arbre de décision")

    col1, col2 = st.columns(2)

    with col1:
        dt_depth_label = st.selectbox(
            "Profondeur maximale - Decision Tree",
            options=["Illimitée", 3, 5, 10, 15, 20],
            index=2,
            key="dt_max_depth"
        )

        dt_max_depth = (
            None if dt_depth_label == "Illimitée"
            else dt_depth_label
        )

    with col2:
        dt_min_split = st.select_slider(
            "Minimum d'échantillons pour diviser un nœud",
            options=[2, 5, 10, 20],
            value=2,
            key="dt_min_samples_split"
        )

    dt_weight_label = st.selectbox(
        "Pondération des classes - Decision Tree",
        options=["Aucune", "Balanced"],
        key="dt_class_weight"
    )


    st.subheader("5.3. K plus proches voisins (KNN)")

    col1, col2, col3 = st.columns(3)

    with col1:
        knn_k = st.slider(
            "Nombre de voisins (n_neighbors)",
            min_value=3,
            max_value=25,
            value=5,
            step=2,
            key="knn_n_neighbors"
        )

    with col2:
        knn_weights = st.selectbox(
            "Pondération des voisins (weights)",
            options=["uniform", "distance"],
            format_func=lambda x: (
                "Uniforme" if x == "uniform"
                else "Selon la distance"
            ),
            key="knn_weights"
        )

    with col3:
        knn_metric = st.selectbox(
            "Distance (metric)",
            options=["minkowski", "euclidean", "manhattan"],
            key="knn_metric"
        )



    if st.button(
        "Entraîner arbre de decision,KNN,régression",
        type="primary",
        key="train_three_models"
    ):

        model_lr = LogisticRegression(
            C=lr_C,
            class_weight=(
                "balanced" if lr_weight_label == "Balanced" else None
            ),
            max_iter=2000,
            random_state=42
        )

        model_dt = DecisionTreeClassifier(
            max_depth=dt_max_depth,
            min_samples_split=dt_min_split,
            class_weight=(
                "balanced" if dt_weight_label == "Balanced" else None
            ),
            random_state=42
        )

        model_knn = KNeighborsClassifier(
            n_neighbors=knn_k,
            weights=knn_weights,
            metric=knn_metric,
            n_jobs=-1
        )

        modeles = {
            "Régression logistique": model_lr,
            "Arbre de décision": model_dt,
            "KNN": model_knn
        }

        resultats = {}
        progression = st.progress(0)
        statut = st.empty()

        try:
            for i, (nom, modele) in enumerate(modeles.items()):

                statut.write(f"Entraînement : {nom}...")

                modele.fit(X_train, y_train)

                predictions = modele.predict(X_test)

                classes = list(modele.classes_)
                index_positif = classes.index(1)

                probabilites = modele.predict_proba(
                    X_test
                )[:, index_positif]

                resultats[nom] = {
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Precision": precision_score(
                        y_test, predictions,
                        pos_label=1, zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test, predictions,
                        pos_label=1, zero_division=0
                    ),
                    "F1-score": f1_score(
                        y_test, predictions,
                        pos_label=1, zero_division=0
                    ),
                    "ROC-AUC": roc_auc_score(y_test, probabilites),
                    "PR-AUC": average_precision_score(
                        y_test, probabilites
                    )
                }

                cle = {
                    "Régression logistique": "lr",
                    "Arbre de décision": "dt",
                    "KNN": "knn"
                }[nom]

                st.session_state[f"{cle}_model"] = modele
                st.session_state[f"{cle}_predictions"] = predictions
                st.session_state[f"{cle}_probabilities"] = probabilites
                st.session_state[f"{cle}_metrics"] = resultats[nom]

                progression.progress((i + 1) / len(modeles))

            st.session_state["three_models_trained"] = True
            statut.success("Les trois modèles ont été entraînés.")

        except Exception as e:
            st.error(f"Erreur pendant l'entraînement : {e}")


    if st.session_state.get("three_models_trained", False):

        st.subheader("5.4. Résultats des trois modèles")

        nouveaux_resultats = {
            "Régression logistique": st.session_state["lr_metrics"],
            "Arbre de décision": st.session_state["dt_metrics"],
            "KNN": st.session_state["knn_metrics"]
        }

        nouveaux_df = pd.DataFrame(nouveaux_resultats).T

        st.dataframe(
            nouveaux_df.style.format("{:.3f}"),
            use_container_width=True
        )

        modele_a_examiner = st.selectbox(
            "Afficher la matrice de confusion de :",
            options=list(nouveaux_resultats.keys()),
            key="modele_confusion"
        )

        cle_selectionnee = {
            "Régression logistique": "lr",
            "Arbre de décision": "dt",
            "KNN": "knn"
        }[modele_a_examiner]

        pred_selectionnees = st.session_state[
            f"{cle_selectionnee}_predictions"
        ]

        cm = confusion_matrix(
            y_test,
            pred_selectionnees,
            labels=[0, 1]
        )

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Prédit : sans panne", "Prédit : panne"],
            y=["Réel : sans panne", "Réel : panne"],
            labels={
                "x": "Classe prédite",
                "y": "Classe réelle",
                "color": "Nombre"
            },
            title=f"Matrice de confusion - {modele_a_examiner}",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig_cm, use_container_width=True)



st.header("Module 6 : Comparaison des cinq modèles")

st.write(
    """
    Cette comparaison rassemble les résultats des modèles entraînés.
    Entraîne chaque modèle avant de comparer ses performances.
    """
)

modeles_comparaison = {
    "Régression logistique": ("lr_metrics", "lr_probabilities"),
    "Arbre de décision": ("dt_metrics", "dt_probabilities"),
    "KNN": ("knn_metrics", "knn_probabilities"),
    "Random Forest": ("rf_metrics", "rf_probabilities"),
    "XGBoost": ("xgb_metrics", "xgb_probabilities")
}

resultats_comparaison = {}
probabilites_disponibles = {}

for nom, (cle_metrics, cle_proba) in modeles_comparaison.items():

    if (
        cle_metrics in st.session_state
        and cle_proba in st.session_state
    ):
        resultats_comparaison[nom] = st.session_state[cle_metrics]
        probabilites_disponibles[nom] = np.asarray(
            st.session_state[cle_proba]
        )

if resultats_comparaison:

    comparaison_df = pd.DataFrame(
        resultats_comparaison
    ).T

    st.subheader("6.1. Tableau comparatif")

    st.dataframe(
        comparaison_df.style.format("{:.3f}"),
        use_container_width=True
    )


    comparaison_longue = (
        comparaison_df
        .reset_index(names="Modèle")
        .melt(
            id_vars="Modèle",
            var_name="Métrique",
            value_name="Score"
        )
    )

    fig_comparaison = px.bar(
        comparaison_longue,
        x="Métrique",
        y="Score",
        color="Modèle",
        barmode="group",
        title="Comparaison des performances",
        hover_data={"Score": ":.3f"}
    )

    st.plotly_chart(
        fig_comparaison,
        use_container_width=True
    )


    if "y_test" in st.session_state:
        y_test_comparaison = pd.Series(
            st.session_state["y_test"]
        ).reset_index(drop=True)

        st.subheader("6.2. Comparaison des courbes ROC")

        fig_roc = go.Figure()

        for nom, proba in probabilites_disponibles.items():

            if len(proba) != len(y_test_comparaison):
                continue

            fpr, tpr, _ = roc_curve(
                y_test_comparaison,
                proba,
                pos_label=1
            )

            auc = roc_auc_score(
                y_test_comparaison,
                proba
            )

            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{nom} (AUC={auc:.3f})"
                )
            )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Aléatoire",
                line=dict(dash="dash")
            )
        )

        fig_roc.update_layout(
            xaxis_title="Taux de faux positifs",
            yaxis_title="Taux de vrais positifs",
            title="Courbes ROC - comparaison des modèles",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig_roc, use_container_width=True)


        st.subheader("6.3. Comparaison des courbes Precision-Recall")

        fig_pr = go.Figure()

        for nom, proba in probabilites_disponibles.items():

            if len(proba) != len(y_test_comparaison):
                continue

            precision_values, recall_values, _ = (
                precision_recall_curve(
                    y_test_comparaison,
                    proba,
                    pos_label=1
                )
            )

            ap = average_precision_score(
                y_test_comparaison,
                proba
            )

            fig_pr.add_trace(
                go.Scatter(
                    x=recall_values,
                    y=precision_values,
                    mode="lines",
                    name=f"{nom} (AP={ap:.3f})"
                )
            )

        fig_pr.update_layout(
            title="Courbes Precision-Recall",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig_pr, use_container_width=True)


    st.subheader("6.4. Meilleur modèle selon chaque métrique")

    for metrique in [
        "Recall", "Precision", "F1-score", "ROC-AUC", "PR-AUC"
    ]:
        if metrique in comparaison_df.columns:
            meilleur = comparaison_df[metrique].idxmax()
            score = comparaison_df.loc[meilleur, metrique]

            st.write(
                f"**{metrique} :** {meilleur} ({score:.3f})"
            )


else:

    st.info(
        "Aucun résultat disponible pour le moment. "
        
        
    )



# ============================================================
# MODULE 7 — EXPLAINABLE AI (XAI)
# ============================================================

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

st.header("7. Explainable AI (XAI)")

# ------------------------------------------------------------
# 1. Vérification des données
# ------------------------------------------------------------

cles_requises = [
    "X_train_resampled",
    "y_train_resampled",
    "X_test",
    "y_test",
]

if not all(cle in st.session_state for cle in cles_requises):
    st.warning(
        "Prépare les données dans le Module 2, puis entraîne "
        "les modèles avant d'utiliser le Module 7."
    )
    st.stop()

X_train_xai = pd.DataFrame(
    st.session_state["X_train_resampled"]
).copy().reset_index(drop=True)

y_train_xai = pd.Series(
    st.session_state["y_train_resampled"]
).reset_index(drop=True)

X_test_xai = pd.DataFrame(
    st.session_state["X_test"]
).copy().reset_index(drop=True)

y_test_xai = pd.Series(
    st.session_state["y_test"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 2. Récupération des modèles
# Vérifier que les clés correspondent à celles des modules 3 à 5
# ------------------------------------------------------------

cles_modeles = {
    "XGBoost": "xgb_model",
    "Random Forest": "rf_model",
    "Régression logistique": "lr_model",
    "Arbre de décision": "dt_model",
    "KNN": "knn_model",
}

modeles_xai = {}

for nom, cle in cles_modeles.items():
    modele = st.session_state.get(cle)
    if modele is not None:
        modeles_xai[nom] = modele

if not modeles_xai:
    st.warning(
        "Aucun modèle entraîné trouvé. "
        "Entraîne un modèle avant de continuer."
    )
    st.stop()

nom_modele_xai = st.selectbox(
    "Choisir le modèle à expliquer",
    list(modeles_xai.keys()),
    key="xai_modele_selectionne",
)

modele_xai = modeles_xai[nom_modele_xai]

# ------------------------------------------------------------
# 3. Nettoyage identique à celui du Module 3 (XGBoost)
# Ne pas modifier le code d'entraînement
# ------------------------------------------------------------

def nettoyer_noms_colonnes(columns):
    noms = []

    for col in columns:
        nom = str(col)
        nom = re.sub(r"[\[\]<]", "_", nom)
        nom = re.sub(r"[^A-Za-z0-9_]", "_", nom)
        noms.append(nom)

    noms_uniques = []
    deja_vus = {}

    for nom in noms:
        if nom in deja_vus:
            deja_vus[nom] += 1
            nom = f"{nom}_{deja_vus[nom]}"
        else:
            deja_vus[nom] = 0

        noms_uniques.append(nom)

    return noms_uniques


X_train_modele = X_train_xai.copy()
X_test_modele = X_test_xai.copy()

if nom_modele_xai == "XGBoost":
    # Même transformation que pendant l'entraînement
    X_train_modele.columns = nettoyer_noms_colonnes(
        X_train_modele.columns
    )
    X_test_modele.columns = nettoyer_noms_colonnes(
        X_test_modele.columns
    )

    # Les noms enregistrés dans le modèle sont la référence.
    # On vérifie les colonnes avant de les réordonner.
    if hasattr(modele_xai, "feature_names_in_"):
        colonnes_modele = list(modele_xai.feature_names_in_)

        if (
            len(colonnes_modele) != len(X_test_modele.columns)
            or set(colonnes_modele) != set(X_test_modele.columns)
        ):
            st.error(
                "Les colonnes XGBoost ne correspondent toujours pas. "
                "Colonnes attendues : "
                + str(colonnes_modele)
                + " | Colonnes obtenues : "
                + str(list(X_test_modele.columns))
            )
            st.stop()

        X_test_modele = X_test_modele[colonnes_modele].copy()
        X_train_modele = X_train_modele[colonnes_modele].copy()

    else:
        # Avec certaines versions de XGBoost, les noms ne sont pas
        # exposés par feature_names_in_. L'ordre doit donc être celui
        # du jeu de données utilisé pendant l'entraînement.
        if len(X_train_modele.columns) != len(X_test_modele.columns):
            st.error(
                "Le nombre de colonnes d'entraînement et de test "
                "ne correspond pas pour XGBoost."
            )
            st.stop()

else:
    # Les autres modèles conservent leurs colonnes d'origine.
    if list(X_train_modele.columns) != list(X_test_modele.columns):
        st.error(
            "Les colonnes d'entraînement et de test ne correspondent "
            "pas pour " + nom_modele_xai + "."
        )
        st.stop()

features_xai = list(X_test_modele.columns)

# ------------------------------------------------------------
# 4. Regroupement des variables catégorielles
# ------------------------------------------------------------

groupes_variables_xai = {}

colonnes_type = [
    col for col in features_xai
    if col == "Type" or col.startswith("Type_")
]

if colonnes_type:
    groupes_variables_xai["Type"] = colonnes_type

for col in features_xai:
    if col not in colonnes_type:
        groupes_variables_xai[col] = [col]


# ------------------------------------------------------------
# 5. Fonction de prédiction
# ------------------------------------------------------------

def probabilite_positive(nom_modele, modele, X):
    X_pred = X.copy()

    if nom_modele == "XGBoost":
        # Les données sont déjà nettoyées et ordonnées.
        # Ne pas appliquer une deuxième transformation ici.
        pass

    if hasattr(modele, "predict_proba"):
        proba = np.asarray(modele.predict_proba(X_pred))

        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]

        return proba[:, 0] if proba.ndim == 2 else proba

    if hasattr(modele, "decision_function"):
        scores = np.asarray(modele.decision_function(X_pred))
        return 1 / (1 + np.exp(-np.clip(scores, -500, 500)))

    return np.asarray(modele.predict(X_pred), dtype=float)


# ------------------------------------------------------------
# 6. Importance native
# ------------------------------------------------------------

st.subheader("Importance native")

if hasattr(modele_xai, "feature_importances_"):
    valeurs = np.asarray(modele_xai.feature_importances_)

    if len(valeurs) == len(features_xai):
        lignes = []

        for variable, colonnes in groupes_variables_xai.items():
            indices = [
                features_xai.index(col)
                for col in colonnes
            ]

            lignes.append({
                "Variable": variable,
                "Importance": float(valeurs[indices].sum()),
            })

        importance_df = pd.DataFrame(lignes).sort_values(
            "Importance", ascending=False
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        affiche = importance_df.sort_values("Importance")
        ax.barh(affiche["Variable"], affiche["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Importance native regroupée")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

elif hasattr(modele_xai, "coef_"):
    coefficients = np.asarray(modele_xai.coef_)

    if coefficients.ndim == 2:
        importance = np.mean(np.abs(coefficients), axis=0)
        coefficients_nets = np.mean(coefficients, axis=0)
    else:
        importance = np.abs(coefficients)
        coefficients_nets = coefficients

    if len(importance) == len(features_xai):
        lignes = []

        for variable, colonnes in groupes_variables_xai.items():
            indices = [
                features_xai.index(col)
                for col in colonnes
            ]

            lignes.append({
                "Variable": variable,
                "Importance": float(importance[indices].sum()),
                "Coefficient net": float(coefficients_nets[indices].sum()),
            })

        importance_df = pd.DataFrame(lignes).sort_values(
            "Importance", ascending=False
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True,
        )

else:
    st.info(
        "Ce modèle ne fournit pas d'importance native. "
        "Utilise la permutation importance ou SHAP."
    )


# ------------------------------------------------------------
# 7. Permutation importance regroupée
# ------------------------------------------------------------

st.subheader("Permutation importance")

if len(X_test_modele) < 2:
    st.warning("Le jeu de test doit contenir au moins deux observations.")

elif st.button(
    "Calculer la permutation importance",
    key="calculer_permutation_xai",
):
    try:
        rng = np.random.RandomState(42)
        n_perm = min(300, len(X_test_modele))

        indices = rng.choice(
            len(X_test_modele),
            size=n_perm,
            replace=False,
        )

        X_perm = X_test_modele.iloc[indices].copy().reset_index(drop=True)
        y_perm = y_test_xai.iloc[indices].reset_index(drop=True)

        score_base = roc_auc_score(
            y_perm,
            probabilite_positive(nom_modele_xai, modele_xai, X_perm),
        )

        lignes = []

        for variable, colonnes in groupes_variables_xai.items():
            scores = []

            for _ in range(5):
                X_melange = X_perm.copy()
                ordre = rng.permutation(len(X_melange))

                X_melange.loc[:, colonnes] = (
                    X_perm.iloc[ordre][colonnes].to_numpy()
                )

                score = roc_auc_score(
                    y_perm,
                    probabilite_positive(
                        nom_modele_xai,
                        modele_xai,
                        X_melange,
                    ),
                )

                scores.append(score)

            differences = score_base - np.asarray(scores)

            lignes.append({
                "Variable": variable,
                "Importance moyenne": float(differences.mean()),
                "Écart-type": float(differences.std()),
            })

        permutation_df = pd.DataFrame(lignes).sort_values(
            "Importance moyenne", ascending=False
        )

        st.session_state["xai_permutation_result"] = (
            nom_modele_xai,
            permutation_df,
        )

    except Exception as e:
        st.error(f"Erreur permutation importance : {e}")

resultat_perm = st.session_state.get("xai_permutation_result")

if resultat_perm is not None and resultat_perm[0] == nom_modele_xai:
    permutation_df = resultat_perm[1]

    st.dataframe(
        permutation_df,
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    affiche = permutation_df.sort_values("Importance moyenne")
    ax.barh(
        affiche["Variable"],
        affiche["Importance moyenne"],
        xerr=affiche["Écart-type"],
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Diminution du ROC-AUC")
    ax.set_title("Permutation importance regroupée")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------
# 8. SHAP regroupé
# ------------------------------------------------------------

st.subheader("SHAP")

def creer_explication_groupee(explication, X_source, groupes):
    import shap

    valeurs = np.asarray(explication.values)

    if valeurs.ndim == 3:
        valeurs = valeurs[:, :, -1]

    if valeurs.ndim != 2:
        raise ValueError(f"Dimensions SHAP non prises en charge : {valeurs.shape}")

    noms_features = list(X_source.columns)
    noms_groupes = list(groupes.keys())

    valeurs_groupees = []

    for variable in noms_groupes:
        indices = [
            noms_features.index(col)
            for col in groupes[variable]
            if col in noms_features
        ]

        valeurs_groupees.append(
            valeurs[:, indices].sum(axis=1)
        )

    valeurs_groupees = np.column_stack(valeurs_groupees)

    donnees_groupees = []

    for variable in noms_groupes:
        colonnes = groupes[variable]

        if variable == "Type" and all(
            col.startswith("Type_") for col in colonnes
        ):
            matrice = X_source[colonnes].to_numpy(dtype=float)
            codes = np.argmax(matrice, axis=1).astype(float)
            codes[np.all(matrice == 0, axis=1)] = -1
            donnees_groupees.append(codes)

        elif len(colonnes) == 1:
            donnees_groupees.append(
                pd.to_numeric(
                    X_source[colonnes[0]],
                    errors="coerce",
                ).to_numpy()
            )

        else:
            donnees_groupees.append(
                X_source[colonnes].mean(axis=1).to_numpy()
            )

    donnees_groupees = np.column_stack(donnees_groupees)

    base = np.asarray(explication.base_values)

    if base.ndim == 0:
        base = np.repeat(base, len(X_source))
    elif base.ndim > 1:
        base = base.reshape(len(X_source), -1)[:, -1]

    return shap.Explanation(
        values=valeurs_groupees,
        base_values=base,
        data=donnees_groupees,
        feature_names=noms_groupes,
    )


if st.button("Calculer les explications SHAP", key="calculer_shap_xai"):
    try:
        import shap

        X_background = X_train_modele.sample(
            n=min(50, len(X_train_modele)),
            random_state=42,
        ).reset_index(drop=True)

        X_expliquer = X_test_modele.sample(
            n=min(100, len(X_test_modele)),
            random_state=42,
        ).reset_index(drop=True)

        def fonction_prediction(data):
            df = pd.DataFrame(data, columns=features_xai)
            return probabilite_positive(nom_modele_xai, modele_xai, df)

        explainer = shap.Explainer(
            fonction_prediction,
            X_background,
            feature_names=features_xai,
        )

        explication_brute = explainer(X_expliquer)

        explication_groupee = creer_explication_groupee(
            explication_brute,
            X_expliquer,
            groupes_variables_xai,
        )

        st.session_state["xai_shap_result"] = (
            nom_modele_xai,
            explication_groupee,
        )

    except Exception as e:
        st.error(f"Erreur SHAP : {e}")

resultat_shap = st.session_state.get("xai_shap_result")

if resultat_shap is not None and resultat_shap[0] == nom_modele_xai:
    try:
        import shap

        explication = resultat_shap[1]
        valeurs = np.asarray(explication.values)

        st.markdown("#### Importance globale SHAP")

        shap.plots.beeswarm(
            explication,
            max_display=len(groupes_variables_xai),
            show=False,
        )
        st.pyplot(plt.gcf())
        plt.close("all")

        importance_shap = pd.DataFrame({
            "Variable": list(explication.feature_names),
            "Importance SHAP moyenne": np.mean(np.abs(valeurs), axis=0),
        }).sort_values("Importance SHAP moyenne", ascending=False)

        st.dataframe(
            importance_shap,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Explication locale")

        index_local = st.selectbox(
            "Choisir une observation",
            options=list(range(len(valeurs))),
            format_func=lambda i: f"Observation {i}",
            key="xai_index_local",
        )

        shap.plots.waterfall(
            explication[index_local],
            max_display=len(groupes_variables_xai),
            show=False,
        )
        st.pyplot(plt.gcf())
        plt.close("all")

        contributions = pd.DataFrame({
            "Variable": list(explication.feature_names),
            "Contribution SHAP": valeurs[index_local],
        })

        contributions["Valeur absolue"] = contributions[
            "Contribution SHAP"
        ].abs()

        contributions = contributions.sort_values(
            "Valeur absolue", ascending=False
        )

        st.dataframe(
            contributions.drop(columns="Valeur absolue"),
            use_container_width=True,
            hide_index=True,
        )

        positives = contributions[
            contributions["Contribution SHAP"] > 0
        ]["Variable"].tolist()

        negatives = contributions[
            contributions["Contribution SHAP"] < 0
        ]["Variable"].tolist()

        if positives:
            st.write(
                "Variables augmentant la sortie expliquée : "
                + ", ".join(positives[:3])
            )

        if negatives:
            st.write(
                "Variables diminuant la sortie expliquée : "
                + ", ".join(negatives[:3])
            )

        st.caption(
            "Le signe indique l'effet sur la sortie expliquée par SHAP, "
            "pas une relation causale."
        )

    except Exception as e:
        st.error(f"Erreur affichage SHAP : {e}")


# ------------------------------------------------------------
# 9. LIME — explication locale
# ------------------------------------------------------------
st.subheader("LIME — Explication locale")

st.write(
    """
    LIME explique une prédiction isolée en apprenant un modèle
    linéaire local autour de l'observation sélectionnée.
    Les contributions sont exprimées dans l'échelle de la
    probabilité de défaillance.
    """
)

# Le résultat LIME est réinitialisé lorsque le modèle change.
if st.session_state.get("lime_modele_courant") != nom_modele_xai:
    st.session_state["lime_modele_courant"] = nom_modele_xai
    st.session_state.pop("xai_lime_result", None)

index_local_lime = st.selectbox(
    "Choisir l'observation à expliquer",
    options=list(range(len(X_test_modele))),
    format_func=lambda i: f"Observation {i}",
    key="lime_index_local",
)

if st.button("Calculer l'explication LIME", key="calculer_lime_xai"):
    try:
        from lime.lime_tabular import LimeTabularExplainer

        classes_xai = list(modele_xai.classes_)
        index_positif_xai = classes_xai.index(1)

        def fonction_prediction_lime(data):
            dataframe = pd.DataFrame(data, columns=features_xai)

            proba = np.asarray(
                modele_xai.predict_proba(dataframe)
            )[:, index_positif_xai]

            return np.column_stack([1 - proba, proba])

        explainer_lime = LimeTabularExplainer(
            training_data=X_train_modele.to_numpy(dtype=float),
            feature_names=features_xai,
            class_names=["Aucune défaillance", "Défaillance"],
            mode="classification",
            discretize_continuous=False,
            random_state=42,
        )

        explication_lime = explainer_lime.explain_instance(
            data_row=X_test_modele.iloc[index_local_lime].to_numpy(
                dtype=float
            ),
            predict_fn=fonction_prediction_lime,
            num_features=len(features_xai),
            num_samples=5000,
            top_labels=1,
        )

        st.session_state["xai_lime_result"] = (
            nom_modele_xai,
            index_local_lime,
            explication_lime,
        )

    except Exception as e:
        st.error(f"Erreur pendant le calcul LIME : {e}")


resultat_lime = st.session_state.get("xai_lime_result")

if resultat_lime is not None and resultat_lime[0] == nom_modele_xai:
    try:
        index_conserve_lime = resultat_lime[1]
        explication_lime = resultat_lime[2]
        label_explique = explication_lime.top_labels[0]

        observation = X_test_modele.iloc[[index_conserve_lime]]

        proba_observee = float(
            probabilite_positive(
                nom_modele_xai,
                modele_xai,
                observation,
            )[0]
        )

        st.markdown(
            f"#### Observation {index_conserve_lime} expliquée par LIME"
        )

        col1, col2 = st.columns(2)
        col1.metric("Défaillance prédite", f"{proba_observee:.2%}")
        col2.metric(
            "Qualité de l'approximation (R²)",
            (
                f"{explication_lime.score:.3f}"
                if explication_lime.score is not None
                else "Non disponible"
            ),
        )

        contributions_lime = pd.DataFrame(
            explication_lime.as_list(label=label_explique),
            columns=["Variable", "Contribution LIME"],
        )

        contributions_lime["Valeur absolue"] = contributions_lime[
            "Contribution LIME"
        ].abs()

        contributions_lime = contributions_lime.sort_values(
            "Valeur absolue",
            ascending=False,
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        affiche_lime = contributions_lime.sort_values("Contribution LIME")
        couleurs_lime = [
            "#c0392b" if valeur < 0 else "#27ae60"
            for valeur in affiche_lime["Contribution LIME"]
        ]

        ax.barh(
            affiche_lime["Variable"],
            affiche_lime["Contribution LIME"],
            color=couleurs_lime,
        )
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Contribution à la probabilité de défaillance")
        ax.set_title("Explication LIME de l'observation")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            contributions_lime.drop(columns="Valeur absolue"),
            use_container_width=True,
            hide_index=True,
        )

        augmentent = contributions_lime[
            contributions_lime["Contribution LIME"] > 0
        ]["Variable"].tolist()

        diminuent = contributions_lime[
            contributions_lime["Contribution LIME"] < 0
        ]["Variable"].tolist()

        st.markdown("#### Interprétation automatique")

        if augmentent:
            st.write(
                "Variables qui augmentent la probabilité de défaillance : "
                + ", ".join(augmentent[:3])
            )

        if diminuent:
            st.write(
                "Variables qui diminuent la probabilité de défaillance : "
                + ", ".join(diminuent[:3])
            )

        st.caption(
            "LIME approxime localement le modèle par un modèle linéaire. "
            "Les contributions portent sur la probabilité de défaillance "
            "et ne prouvent pas une relation causale."
        )

    except Exception as e:
        st.error(f"Erreur pendant l'affichage de l'explication LIME : {e}")
