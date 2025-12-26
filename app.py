import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---- SAYFA AYARLARI ----
st.set_page_config(
    page_title="KNN & K-Means Görselleştirici",
    page_icon="📊",
    layout="wide"
)

# ---- ŞIK ÜST BAŞLIK ----
st.markdown(
    """
    <style>
    .big-title {
        font-size: 32px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 16px;
        text-align: center;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    .step-box {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        background-color: #f5f5f5;
        margin-bottom: 0.5rem;
        font-size: 14px;
    }
    </style>
    <div class="big-title">KNN & K-Means - Adım Adım Görselleştirici</div>
    <div class="sub-title">k-en yakın komşu (KNN) ve k-ortalama (K-Means) algoritmalarının mantığını görselleştir.</div>
    """,
    unsafe_allow_html=True
)

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Ayarlar")

algo = st.sidebar.radio(
    "Algoritma Seç:",
    ["KNN (Sınıflandırma)", "K-Means (K-Ortalamalar - Kümeleme)"]
)

np.random.seed(42)

# ---- YARDIMCI FONKSİYONLAR ----

def generate_classification_data(n_per_class=30):
    """2 sınıflı 2D yapay veri üretir."""
    mean1 = [1, 1]
    mean2 = [4, 4]
    cov = [[0.5, 0], [0, 0.5]]

    class1 = np.random.multivariate_normal(mean1, cov, n_per_class)
    class2 = np.random.multivariate_normal(mean2, cov, n_per_class)

    X = np.vstack([class1, class2])
    y = np.array([0]*n_per_class + [1]*n_per_class)
    return X, y

def knn_predict(X_train, y_train, x_new, k):
    """KNN mantığını uygular, adım adım için detay döner."""
    # 1. Mesafeleri hesapla
    distances = np.linalg.norm(X_train - x_new, axis=1)

    # 2. En küçük k taneyi seç
    idx_sorted = np.argsort(distances)
    k_idx = idx_sorted[:k]

    # 3. Komşu etiketleri
    k_labels = y_train[k_idx]

    # 4. Çoğunluk oyu
    counts = np.bincount(k_labels)
    pred_label = np.argmax(counts)

    return pred_label, distances, k_idx, k_labels

def generate_clustering_data(n_points=80):
    """K-Means için 3 bulut veri üretir."""
    mean1 = [1, 1]
    mean2 = [5, 1]
    mean3 = [3, 4]
    cov = [[0.4, 0], [0, 0.4]]

    c1 = np.random.multivariate_normal(mean1, cov, n_points // 3)
    c2 = np.random.multivariate_normal(mean2, cov, n_points // 3)
    c3 = np.random.multivariate_normal(mean3, cov, n_points - 2*(n_points // 3))

    X = np.vstack([c1, c2, c3])
    return X

def kmeans_step_by_step(X, k, max_iter=10):
    """
    K-Means'i adım adım çalıştırır.
    Her iterasyondaki merkezleri ve atamaları kaydeder.
    """
    n_samples = X.shape[0]
    # Başlangıç merkezlerini rastgele seç
    init_idx = np.random.choice(n_samples, k, replace=False)
    centers = X[init_idx]

    history = []  # (centers, labels) listesi

    for it in range(max_iter):
        # 1. Her noktayı en yakın merkeze ata
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        history.append((centers.copy(), labels.copy()))

        # 2. Merkezleri güncelle
        new_centers = []
        for ci in range(k):
            cluster_points = X[labels == ci]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(axis=0))
            else:
                # Boş küme olursa merkez değişmesin
                new_centers.append(centers[ci])
        new_centers = np.vstack(new_centers)

        # Değişim yoksa erken durdur
        if np.allclose(new_centers, centers):
            centers = new_centers
            history.append((centers.copy(), labels.copy()))
            break

        centers = new_centers

    return history


# ---- KNN ARAYÜZÜ ----
if algo == "KNN (Sınıflandırma)":
    st.sidebar.subheader("🔵 KNN Ayarları")

    n_per_class = st.sidebar.slider(
        "Her sınıf için nokta sayısı", 10, 100, 40, step=5
    )
    k_value = st.sidebar.slider(
        "k (komşu sayısı)", 1, 15, 5, step=2
    )

    # Yeni nokta koordinatları
    x_new_x = st.sidebar.slider("Yeni nokta X", -1.0, 6.0, 2.5, step=0.1)
    x_new_y = st.sidebar.slider("Yeni nokta Y", -1.0, 6.0, 2.5, step=0.1)
    x_new = np.array([x_new_x, x_new_y])

    X, y = generate_classification_data(n_per_class)

    pred_label, distances, k_idx, k_labels = knn_predict(X, y, x_new, k_value)

    col_plot, col_text = st.columns([2, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(5, 5))

        # Sınıfları farklı renklerde çiz
        ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Sınıf 0", alpha=0.7)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Sınıf 1", alpha=0.7)

        # Komşular
        ax.scatter(
            X[k_idx, 0], X[k_idx, 1],
            s=120, edgecolor="black", facecolor="none", linewidths=1.5,
            label=f"En yakın {k_value} komşu"
        )

        # Yeni nokta
        ax.scatter(
            x_new[0], x_new[1],
            marker="*", s=250,
            c="red" if pred_label == 1 else "blue",
            label="Yeni nokta"
        )

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("KNN - Veri Noktaları ve Komşular")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    with col_text:
        st.markdown("### 🧠 KNN Adım Adım Mantık")
        st.markdown(
            f"""
            <div class="step-box">
            <b>1. Adım – Mesafeleri Hesapla:</b><br>
            Yeni nokta ile eğitim verisindeki her nokta arasındaki Öklid mesafesi hesaplanır.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="step-box">
            <b>2. Adım – En Yakın k Noktayı Seç:</b><br>
            Mesafeler küçükten büyüğe sıralanır ve en küçük <b>{k_value}</b> tanesi komşu olarak seçilir.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Komşu etiket sayıları
        unique, counts = np.unique(k_labels, return_counts=True)
        count_dict = dict(zip(unique, counts))
        class0_count = count_dict.get(0, 0)
        class1_count = count_dict.get(1, 0)

        st.markdown(
            f"""
            <div class="step-box">
            <b>3. Adım – Çoğunluk Oyu:</b><br>
            Komşuların etiketleri sayılır:<br>
            • Sınıf 0: {class0_count} komşu<br>
            • Sınıf 1: {class1_count} komşu<br><br>
            Çoğunlukta olan etiket, yeni noktanın sınıfı olur.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="step-box">
            <b>Sonuç:</b><br>
            Yeni nokta <b>Sınıf {pred_label}</b> olarak sınıflandırıldı.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📜 Kısa Özet"):
            st.write(
                "KNN, etiketli veride yeni bir noktanın sınıfını belirlemek için en yakın k komşuya bakar ve çoğunluk oyu ile karar verir."
            )

# ---- K-MEANS ARAYÜZÜ ----
else:
    st.sidebar.subheader("🟣 K-Means Ayarları")

    n_points = st.sidebar.slider(
        "Toplam nokta sayısı", 30, 200, 90, step=10
    )
    k_clusters = st.sidebar.slider(
        "Küme sayısı (k)", 2, 6, 3, step=1
    )

    max_iter = st.sidebar.slider(
        "Maksimum iterasyon", 1, 15, 8, step=1
    )

    X = generate_clustering_data(n_points)
    history = kmeans_step_by_step(X, k_clusters, max_iter=max_iter)

    # Kullanıcının göreceği iterasyon
    it_step = st.sidebar.slider(
        "Gösterilecek iterasyon", 1, len(history), 1, step=1
    ) - 1

    centers_step, labels_step = history[it_step]

    col_plot, col_text = st.columns([2, 1])

    with col_plot:
        fig, ax = plt.subplots(figsize=(5, 5))

        # Her kümeyi farklı renkte çiz (matplotlib kendi renk paletini kullanır)
        for ci in range(k_clusters):
            cluster_points = X[labels_step == ci]
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.7, label=f"Küme {ci}")

        # Merkezler
        ax.scatter(
            centers_step[:, 0],
            centers_step[:, 1],
            marker="X", s=250,
            edgecolor="black",
            linewidths=1.5,
            label="Merkezler"
        )

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title(f"K-Means - {it_step+1}. İterasyon")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    with col_text:
        st.markdown("### 🧠 K-Means Adım Adım Mantık")

        st.markdown(
            """
            <div class="step-box">
            <b>1. Adım – Başlangıç Merkezleri:</b><br>
            Veriden rastgele k nokta seçilir ve başlangıç küme merkezleri olarak atanır.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="step-box">
            <b>2. Adım – Noktaları En Yakın Merkeze Ata:</b><br>
            Her nokta için tüm merkezlere olan mesafe hesaplanır ve en yakın merkeze göre <b>küme etiketi</b> verilir.
            Bu adım, şu an gösterilen <b>{it_step+1}. iterasyonda</b> yapılan atamaları içeriyor.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="step-box">
            <b>3. Adım – Merkezleri Güncelle:</b><br>
            Her kümedeki noktaların ortalaması alınır ve bu ortalama, yeni küme merkezi olur 
            (bu yüzden adı <b>k-ortalama</b>).
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="step-box">
            <b>4. Adım – Tekrarla:</b><br>
            Atama ve güncelleme adımları değişim kalmayana kadar veya maksimum iterasyona ulaşana kadar tekrarlanır.
            Şu an toplam <b>{len(history)}</b> adım kaydedildi.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📜 Kısa Özet"):
            st.write(
                "K-Means, veriyi k tane küme olacak şekilde böler. Her kümenin merkezi, o kümedeki noktaların ortalamasıdır. "
                "Amaç, noktalar ile kendi merkezleri arasındaki mesafelerin toplamını en aza indirmektir."
            )

