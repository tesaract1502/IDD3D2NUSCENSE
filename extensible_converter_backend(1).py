class IDD3DInstanceConverter(BaseConverter):
    """Generate nuScenes instance.json from label files - SIMPLIFIED VERSION"""
    
    def __init__(self):
        super().__init__("instance")
    
    def run(self, dataloader: 'IDD3DDataLoader', loghandler: 'LogHandler'):
        import uuid
        
        # Read category.json to get category tokens
        categorypath = os.path.join(dataloader.annotout, 'category.json')
        categorymap = {}  # categoryname -> token
        if os.path.exists(categorypath):
            try:
                with open(categorypath, 'r') as f:
                    categories = json.load(f)
                    for cat in categories:
                        categorymap[cat['name']] = cat['token']
            except Exception:
                pass
        
        # IDD3D to nuScenes category mapping
        idd3d_to_nuscenes_categories = {
            'Car': 'vehicle.car',
            'Truck': 'vehicle.truck',
            'Bus': 'vehicle.bus',
            'Motorcycle': 'vehicle.motorcycle',
            'Bicycle': 'vehicle.bicycle',
            'Auto': 'vehicle.auto',
            'Person': 'human.pedestrian.adult',
            'Rider': 'human.pedestrian.rider',
            'Animal': 'animal',
            'TrafficLight': 'static_object.trafficlight',
            'TrafficSign': 'static_object.trafficsign',
            'Pole': 'static_object.pole',
            'OtherVehicle': 'vehicle.other',
            'Misc': 'movable_object.debris'
        }
        
        # Read annotdata.json to get frame information
        annotdata = dataloader.read_annotations()
        if not annotdata:
            loghandler.log("No annotations found", "warning")
            return
        
        frameids = sorted(annotdata.keys())
        
        # Track unique instances across all frames
        instance_tracker = {}  # objid -> {instancetoken, categorytoken}
        
        # First pass: collect all unique objects
        for frameid in frameids:
            labelpath = os.path.join(dataloader.labeldir, f'{frameid}.json')
            if not os.path.exists(labelpath):
                continue
            
            try:
                with open(labelpath, 'r') as f:
                    labelobjects = json.load(f)
                
                for obj in labelobjects:
                    objid = obj.get('obj_id')
                    objtype = obj.get('obj_type')
                    
                    if not objid or not objtype:
                        continue
                    
                    if objid not in instance_tracker:
                        # Generate instance token for this unique object
                        instancetoken = uuid.uuid4().hex
                        
                        # Get category token
                        categoryname = idd3d_to_nuscenes_categories.get(
                            objtype, 
                            f'movable_object.{objtype.lower()}'
                        )
                        categorytoken = categorymap.get(categoryname, uuid.uuid4().hex)
                        
                        instance_tracker[objid] = {
                            'instancetoken': instancetoken,
                            'categorytoken': categorytoken
                        }
            
            except Exception as e:
                loghandler.log(f"Error processing label {frameid}: {str(e)}", "warning")
        
        # Create instance entries (SIMPLIFIED - only token and category_token)
        instances = []
        for objid, data in instance_tracker.items():
            instance = {
                'token': data['instancetoken'],
                'category_token': data['categorytoken']
            }
            instances.append(instance)
        
        # Save instance.json
        outpath = os.path.join(dataloader.annotout, 'instance.json')
        with open(outpath, 'w') as f:
            json.dump(instances, f, indent=2)
        
        loghandler.log(f"Instance file created with {len(instances)} instances", "success")
