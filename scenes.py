class IDD3DSceneConverter:
    """Convert IDD3D sequence to nuScenes scene.json format"""
    
    def __init__(self, sequence_name='seq_10'):
        self.sequence_name = sequence_name
        self.scene_token = uuid.uuid4().hex
        self.log_token = uuid.uuid4().hex
    
    def run(self, data_loader, log_handler):
        """
        Generate scene.json by iterating through all frames
        
        Args:
            data_loader: IDD3DDataLoader instance
            log_handler: LogHandler instance for logging
        """
        
        # Load annot_data.json to get all frame IDs
        annot_data = data_loader.read_annotations()
        if not annot_data:
            log_handler.log("⚠ No annotations found for scene conversion", 'warning')
            return
        
        # Get sorted list of frame IDs
        frame_ids = sorted(annot_data.keys())
        if not frame_ids:
            log_handler.log("⚠ No frames found", 'warning')
            return
        
        log_handler.log(f"Found {len(frame_ids)} frames in sequence", 'info')
        
        # Generate scene object
        scene = {
            "token": self.scene_token,
            "log_token": self.log_token,
            "nbr_samples": len(frame_ids),
            "first_sample_token": frame_ids[0],  # Will be updated with actual sample token
            "last_sample_token": frame_ids[-1],   # Will be updated with actual sample token
            "name": f"scene-{self.sequence_name}",
            "description": f"IDD3D sequence {self.sequence_name} with {len(frame_ids)} frames"
        }
        
        # Generate log entry
        log_entry = {
            "token": self.log_token,
            "logfile": f"idd3d_{self.sequence_name}",
            "vehicle": "IDD3D_vehicle",
            "date_captured": "2022-01-18",  # From sequence name 20220118103308
            "location": "Hyderabad, India"
        }
        
        # Save scene.json
        scene_path = os.path.join(data_loader.annot_out, 'scene.json')
        with open(scene_path, 'w') as f:
            json.dump([scene], f, indent=2)
        
        # Save log.json
        log_path = os.path.join(data_loader.annot_out, 'log.json')
        with open(log_path, 'w') as f:
            json.dump([log_entry], f, indent=2)
        
        log_handler.log(f"✓ Scene file created: {len(frame_ids)} samples", 'success')
        log_handler.log(f"  Scene token: {self.scene_token}", 'info')
        log_handler.log(f"  Log token: {self.log_token}", 'info')
        
        # Store tokens for other converters to use
        self.first_sample_token = frame_ids[0]
        self.last_sample_token = frame_ids[-1]
        
        return {
            'scene_token': self.scene_token,
            'log_token': self.log_token,
            'num_samples': len(frame_ids),
            'first_frame_id': frame_ids[0],
            'last_frame_id': frame_ids[-1]
        }
