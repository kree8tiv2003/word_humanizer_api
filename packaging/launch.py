"""PyInstaller entry point for the Script Studio desktop app."""
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    from script_studio.desktop import main
    main()
