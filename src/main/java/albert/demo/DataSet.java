package albert.demo;

import org.apache.commons.io.IOUtils;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Optional;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * <a href="https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection">SMS Spam Collection のデータセット</a> を
 * UCI Machine Learning Repository からダウンロードする機能を提供します。
 *
 * @author KOMIYA Atsushi
 */
public class DataSet {
    public static class SMSSpamCollection {
        static final String URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip";
        static final String ZIP_FILENAME = "SMSSpamCollection.zip";
        static final String DATA_FILENAME = "SMSSpamCollection";

        public static void prepareIfNeed() {
            download();
            expand();
        }

        static void download() {
            if (new File(ZIP_FILENAME).exists()) {
                return;
            }

            HttpURLConnection conn = null;
            try {
                URL url = new URL(URL);
                conn = (HttpURLConnection) url.openConnection();
                try (InputStream is = conn.getInputStream();
                     OutputStream os = new FileOutputStream(ZIP_FILENAME)) {
                    IOUtils.copy(is, os);
                }

            } catch (IOException e) {
                throw new RuntimeException(e);

            } finally {
                if (conn != null) {
                    conn.disconnect();
                }
            }
        }

        static void expand() {
            if (new File(DATA_FILENAME).exists()) {
                return;
            }

            try (ZipFile zipFile = new ZipFile(new File(ZIP_FILENAME))) {
                Optional<? extends ZipEntry> entry = zipFile.stream()
                        .filter(e -> DATA_FILENAME.equals(e.getName()))
                        .findFirst();

                entry.ifPresent(e -> {
                    try (InputStream is = zipFile.getInputStream(e);
                         OutputStream os = new FileOutputStream(DATA_FILENAME)) {
                        IOUtils.copy(is, os);

                    } catch (IOException ex) {
                        throw new RuntimeException(ex);
                    }
                });

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}