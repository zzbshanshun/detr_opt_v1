import os, pdb
import matplotlib.pyplot as plt
import json

def get_log_dict(f_path):
    epoch_infos = None
    with open(f_path, 'r') as file:
        epoch_infos = file.readlines()
    if epoch_infos is None:
        print("read log : {} error!".format(f_path))
        return {}
    else:
        print("> get {} epoch number is {}.".format(f_path, len(epoch_infos)))
    
    x_eidx = []
    y_train_loss = []
    y_test_loss = []
    y_coco_eval = []
    for ei, einfo in enumerate(epoch_infos):
        data = json.loads(einfo)
        y_train_loss.append(data["train_loss"])
        y_test_loss.append(data["test_loss"])
        y_coco_eval.append(data["test_coco_eval_bbox"][0])
        x_eidx.append(ei+1)
    data_dict = {"x_eidx":x_eidx, "train_loss": y_train_loss, 
                "test_loss":y_test_loss, "coco_eval": y_coco_eval}
    return data_dict

def plot_info(tdata_dict : dict, output_dir):
    for key in tdata_dict:
        curve_name=os.path.splitext(key)[0]
        
        fig_names = ["train_loss", "test_loss", "coco_eval"]
        for fi, fig_name in enumerate(fig_names):
            plt.figure(fi)
            x=tdata_dict[key]["x_eidx"]
            y=tdata_dict[key][fig_name]
            plt.plot(x,y,marker='', label=curve_name)
            plt.grid(which='major', axis="x",ls='--',linewidth=0.5,color='k')
            plt.minorticks_on()
            plt.grid(which="minor", linestyle=":", linewidth=0.4, color="lightgray")
            plt.legend()
            plt.title(fig_name+" curve")
            save_path = os.path.join(output_dir, fig_name+".png")
            plt.savefig(save_path, dpi=400)
    print("> save results to {}.".format(output_dir))

def main(log_dir: str, output_dir):
    log_files = os.listdir(log_dir)
    print("> get train log files {} .".format(len(log_files)))
    tdata_dict = {}
    for f in log_files:
        f_path = os.path.join(log_dir, f)
        # get epoch infos 
        tdata_dict[f] = get_log_dict(f_path)
    
    # plot
    plot_info(tdata_dict, output_dir)
    return

if __name__ == "__main__":
    log_dir = "./logs"
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    main(log_dir, output_dir)