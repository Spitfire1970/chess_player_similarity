import os
import torch
from encoder.model import Encoder
 
class EndpointHandler():
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(os.path.join(model_dir, "6.pt"), self.device, weights_only=True)
        self.model = Encoder(self.device)
        state_dict = checkpoint['model_state']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
 
    def __call__(self, data):
        data = data.get("inputs", data)
        if data["length"] == 0:
            print('entering test endpoint')
            print('exiting test endpoint')
            return {"reply": "hello from inference api!!"}
        tensor = torch.tensor(data["tensor"]).float().to(self.device)
        if data["length"] == 1:
            print('entering ai_move endpoint')
            with torch.no_grad():
                embed = self.model(tensor)
                embed = embed / torch.norm(embed)
            print('exiting ai_move endpoint')
            return {"reply": embed.cpu().numpy().tolist()}
        else:
            print('entering create_username endpoint')
            with torch.no_grad():
                embeds = self.model(tensor)
                embeds = embeds.view((1, data["num_games"], -1)).to(self.device)
                centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
                centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)
            centroids_incl = centroids_incl.cpu().squeeze(1)
            final_embeds = centroids_incl[0].numpy().tolist()
            print('exiting create_username endpoint')
            return {"reply": final_embeds}