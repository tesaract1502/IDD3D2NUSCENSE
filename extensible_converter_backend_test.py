class IDD3DCameraConverter(BaseConverter):
    """Convert IDD3D camera images from PNG to JPEG - keeps cam0-cam5 naming"""
    
    def __init__(self):
        super().__init__("camera")
    
    def run(self, dataloader: 'IDD3DDataLoader', loghandler: 'LogHandler'):
        try:
            from PIL import Image
            usepil = True
        except ImportError:
            usepil = False
            loghandler.log("PIL/Pillow not available, skipping camera conversion", "warning")
            return
        
        cameradir = os.path.join(dataloader.seqbase, "camera")
        if not os.path.exists(cameradir):
            loghandler.log("No camera directory found", "warning")
            return
        
        # Keep original camera naming - NO MAPPING NEEDED
        camerachannels = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"]
        
        sweepsdir = os.path.join(dataloader.outdata, "sweeps")
        converted = 0
        errors = 0
        
        for camid in camerachannels:
            camfolder = os.path.join(cameradir, camid)
            if not os.path.exists(camfolder):
                continue
            
            # FIXED: Use camid directly instead of nuscenescamname variable
            sweepcamdir = os.path.join(sweepsdir, camid)
            os.makedirs(sweepcamdir, exist_ok=True)
            
            pngfiles = sorted([f for f in os.listdir(camfolder) if f.lower().endswith('.png')])
            
            for fname in pngfiles:
                srcpath = os.path.join(camfolder, fname)
                basename = os.path.splitext(fname)[0]
                dstpath = os.path.join(sweepcamdir, basename + '.jpg')
                
                try:
                    if usepil:
                        img = Image.open(srcpath)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(dstpath, 'JPEG', quality=95)
                        converted += 1
                except Exception as e:
                    errors += 1
                    loghandler.log(f"Error converting {fname}: {str(e)}", "error")
        
        loghandler.log(f"Camera conversion complete: {converted} images converted to sweeps, {errors} errors", "success")
