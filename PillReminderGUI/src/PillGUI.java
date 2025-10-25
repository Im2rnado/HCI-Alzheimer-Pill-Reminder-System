import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.Socket;
import java.text.SimpleDateFormat;
import java.util.Date;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;

public class PillGUI {

    private JFrame frame;
    private JPanel mainPanel;
    private JLabel statusLabel;
    private JLabel timeLabel;

    // Colors
    private final Color COLOR_NEUTRAL = new Color(220, 220, 220); // Light gray
    private final Color COLOR_REMINDER = new Color(255, 230, 153); // Light yellow
    private final Color COLOR_SUCCESS = new Color(204, 255, 204); // Light green
    private final Color COLOR_WARNING = new Color(255, 179, 179); // Light red

    // Socket
    private static final String HOST = "localhost";
    private static final int PORT = 65432;

    public static void main(String[] args) {
        // Run the GUI on the Event Dispatch Thread
        SwingUtilities.invokeLater(() -> new PillGUI().createAndShowGUI());
    }

    private void createAndShowGUI() {
        frame = new JFrame("Pill Reminder System");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(600, 300);

        mainPanel = new JPanel(new BorderLayout());
        mainPanel.setBackground(COLOR_NEUTRAL);
        mainPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        // Time Label (Top)
        timeLabel = new JLabel();
        timeLabel.setFont(new Font("Arial", Font.PLAIN, 24));
        timeLabel.setHorizontalAlignment(SwingConstants.CENTER);
        updateTime(); // Set initial time
        mainPanel.add(timeLabel, BorderLayout.NORTH);

        // Status Label (Center)
        statusLabel = new JLabel("Connecting to system...");
        statusLabel.setFont(new Font("Arial", Font.BOLD, 32));
        statusLabel.setHorizontalAlignment(SwingConstants.CENTER);
        mainPanel.add(statusLabel, BorderLayout.CENTER);

        frame.getContentPane().add(mainPanel);
        frame.setLocationRelativeTo(null); // Center on screen
        frame.setVisible(true);

        // Start the clock timer (updates every second)
        new Timer(1000, e -> updateTime()).start();

        // Start the socket listener thread
        startSocketListener();
    }

    private void updateTime() {
        SimpleDateFormat sdf = new SimpleDateFormat("EEE, MMM d, yyyy --- hh:mm:ss a");
        timeLabel.setText(sdf.format(new Date()));
    }

    private void startSocketListener() {
        // Run socket communication in a separate thread to not freeze the GUI
        Thread socketThread = new Thread(() -> {
            try (Socket socket = new Socket(HOST, PORT);
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
                
                String serverMessage;
                while ((serverMessage = in.readLine()) != null) {
                    // Parse the JSON-like message
                    // Using simple string parsing to avoid external libs
                    String event = getStringField(serverMessage, "event");
                    String message = getStringField(serverMessage, "message");

                    // Update GUI safely on the Event Dispatch Thread
                    SwingUtilities.invokeLater(() -> updateGUI(event, message));
                }

            } catch (Exception e) {
                e.printStackTrace();
                SwingUtilities.invokeLater(() -> updateGUI("ERROR", "Connection lost. Please restart."));
            }
        });
        socketThread.setDaemon(true); // Ensure thread exits when app closes
        socketThread.start();
    }

    // Simple helper to parse our specific JSON string
    private String getStringField(String json, String field) {
        try {
            String key = "\"" + field + "\": \"";
            int start = json.indexOf(key) + key.length();
            int end = json.indexOf("\"", start);
            return json.substring(start, end);
        } catch (Exception e) {
            return ""; // Not found
        }
    }

    private void updateGUI(String event, String message) {
        statusLabel.setText("<html><div style='text-align: center;'>" + message + "</div></html>");

        switch (event) {
            case "SUCCESS":
                mainPanel.setBackground(COLOR_SUCCESS);
                break;
            case "WARNING":
                mainPanel.setBackground(COLOR_WARNING);
                playSound("alert.mp3"); // Play warning sound
                break;
            case "REMINDER":
                mainPanel.setBackground(COLOR_REMINDER);
                break;
            case "STATUS":
                mainPanel.setBackground(COLOR_NEUTRAL);
                break;
            case "ERROR":
                mainPanel.setBackground(COLOR_WARNING);
                statusLabel.setText("CONNECTION ERROR");
                break;
            default:
                mainPanel.setBackground(COLOR_NEUTRAL);
        }
    }

    public synchronized void playSound(final String url) {
        // Run in a new thread to not block GUI updates
        new Thread(() -> {
            try {
                File soundFile = new File(url);
                if (!soundFile.exists()) {
                    System.err.println("Warning: Sound file not found: " + url);
                    return;
                }
                AudioInputStream audioIn = AudioSystem.getAudioInputStream(soundFile.toURI().toURL());
                Clip clip = AudioSystem.getClip();
                clip.open(audioIn);
                clip.start();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }
}