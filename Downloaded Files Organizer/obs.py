def watcher(path):
    #python script to observe changes in a folder
    import sys
    import time
    import os
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    from move_to_directory import add_to_dir
    
    
    class Handler(FileSystemEventHandler):
        def on_created(self,event):
            if event.event_type=="created":
                    file_name = os.path.basename(event.src_path)
                    ext = os.path.splitext(event.src_path)[1]
                    time.sleep(2)
                    add_to_dir(ext[1:],event.src_path,path)
                    observer.stop()



    observer = Observer()
    event_handler   = Handler()
    observer.schedule(event_handler,path,recursive=True)
    observer.start()
    observer.join()

